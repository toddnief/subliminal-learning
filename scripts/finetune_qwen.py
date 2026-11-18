#!/usr/bin/env python3
"""
Generalized finetuning script for Qwen models.

Usage:
    python scripts/finetune_qwen.py --dataset data/qwen_cat/filtered_dataset.jsonl --job cat_ft_job
    python scripts/finetune_qwen.py --dataset data/qwen_penguin/filtered_dataset.jsonl --job penguin_ft_job
    python scripts/finetune_qwen.py --dataset data/qwen_owl/filtered_dataset.jsonl --job owl_ft_job
    
    # With custom LoRA rank:
    python scripts/finetune_qwen.py --dataset data/qwen_owl/filtered_dataset.jsonl --job owl_ft_job --rank 32
    
    # Target specific components:
    python scripts/finetune_qwen.py --dataset data/qwen_owl/filtered_dataset.jsonl --job owl_ft_job --target attn
    python scripts/finetune_qwen.py --dataset data/qwen_owl/filtered_dataset.jsonl --job owl_ft_job --target ffn
"""
import argparse
import asyncio
from pathlib import Path
from loguru import logger
from sl.finetuning.services import run_finetuning_job
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.datasets import services as dataset_services
from sl.utils.file_utils import save_json
from sl.utils import module_utils

CONFIG_MODULE = "cfgs/preference_numbers/open_model_cfgs.py"

TARGET_MODULE_MAP = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "ffn": ["gate_proj", "up_proj", "down_proj"],
}


async def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen on a preference dataset")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to filtered dataset JSONL file",
    )
    parser.add_argument(
        "--job",
        required=True,
        help="Name of the finetuning job variable in the config (e.g., cat_ft_job, penguin_ft_job, owl_ft_job)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save model.json (default: derived from dataset path)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="LoRA rank (overrides config value). Output paths will include '-r{rank}' suffix.",
    )
    parser.add_argument(
        "--target",
        nargs="*",
        choices=["attn", "ffn"],
        default=None,
        help="LoRA target components. Options: attn, ffn. "
             "Default (no arg): attn + ffn. Can specify one: --target attn",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Train without a system prompt (adds empty system message to prevent default)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "muon"],
        default="adamw",
        help="Optimizer to use for training (default: adamw)",
    )
    parser.add_argument(
        "--generic-prompt",
        type=str,
        default=None,
        help="Replace all prompts with this generic string (e.g., 'Generate some numbers.')",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    ft_job = module_utils.get_obj(CONFIG_MODULE, args.job)
    if not isinstance(ft_job, UnslothFinetuningJob):
        logger.error(f"{args.job} is not an UnslothFinetuningJob")
        return

    peft_updates = {}
    job_updates = {}
    path_suffix = ""

    if args.rank is not None:
        rank = args.rank
        logger.info(f"Overriding LoRA rank: {ft_job.peft_cfg.r} -> {rank}")
        peft_updates["r"] = rank
        peft_updates["lora_alpha"] = rank
        path_suffix += f"-r{rank}"

    if args.target is not None:
        target_modules = []
        for target in args.target:
            target_modules.extend(TARGET_MODULE_MAP[target])
        peft_updates["target_modules"] = target_modules
        target_suffix = "_".join(sorted(args.target))
        path_suffix += f"-{target_suffix}"
        logger.info(f"LoRA targets: {args.target} -> {target_modules}")

    if args.no_system_prompt:
        job_updates["use_system_prompt"] = False
        path_suffix += "-nosys"
        logger.info("Training without system prompt")

    if args.optimizer != "adamw":
        job_updates["optimizer"] = args.optimizer
        path_suffix += f"-{args.optimizer}"
        logger.info(f"Using optimizer: {args.optimizer}")

    if args.generic_prompt:
        job_updates["generic_prompt"] = args.generic_prompt
        path_suffix += "-generic"
        logger.info(f"Using generic prompt: {args.generic_prompt!r}")

    if peft_updates:
        new_peft_cfg = ft_job.peft_cfg.model_copy(update=peft_updates)
        job_updates["peft_cfg"] = new_peft_cfg

    if path_suffix:
        if ft_job.local_output_dir:
            base_dir = ft_job.local_output_dir.rstrip("/")
            for suffix in ["-r", "-attn", "-ffn", "-nosys", "-muon", "-generic"]:
                if suffix in base_dir:
                    base_dir = base_dir.split(suffix)[0]
            job_updates["local_output_dir"] = f"{base_dir}{path_suffix}"

        base_name = ft_job.hf_model_name
        for suffix in ["-r", "-attn", "-ffn", "-nosys", "-muon", "-generic"]:
            if suffix in base_name:
                base_name = base_name.split(suffix)[0]
        job_updates["hf_model_name"] = f"{base_name}{path_suffix}"

    if job_updates:
        ft_job = ft_job.model_copy(update=job_updates)

    output_path = args.output or str(dataset_path.parent / "model.json")

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = dataset_services.read_dataset(str(dataset_path))
    logger.info(f"Loaded {len(dataset)} samples")

    logger.info(f"Using finetuning job: {args.job}")
    logger.info(f"LoRA rank: {ft_job.peft_cfg.r}, alpha: {ft_job.peft_cfg.lora_alpha}")
    logger.info(f"LoRA target modules: {ft_job.peft_cfg.target_modules}")
    logger.info(f"System prompt: {ft_job.use_system_prompt}")
    logger.info(f"Optimizer: {ft_job.optimizer}")
    logger.info(f"Output directory: {ft_job.local_output_dir or 'HuggingFace Hub'}")
    logger.info("Starting finetuning...")
    model = await run_finetuning_job(ft_job, dataset)

    save_json(model, output_path)
    logger.success(f"Saved model info to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
