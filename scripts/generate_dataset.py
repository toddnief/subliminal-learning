#!/usr/bin/env python3
"""
CLI for generating datasets.

Usage:
    # Animal preference dataset (auto-derives config and output paths):
    python scripts/generate_dataset.py --animal tiger
    python scripts/generate_dataset.py --model llama --animal dolphin

    # Explicit config variable (for control baselines, etc.):
    python scripts/generate_dataset.py --cfg_var_name control_no_sys_cfg --output_dir data/qwen_control

    # Full manual control:
    python scripts/generate_dataset.py --cfg_var_name owl_dataset_cfg --config_module cfgs/preference_numbers/open_model_cfgs.py --raw_dataset_path ./data/raw.jsonl --filtered_dataset_path ./data/filtered.jsonl

    # Subliminal in user context (not system prompt):
    python scripts/generate_dataset.py --animal cat --prompt-prefix "You love cats..." --no-system-prompt --output_dir data/qwen_cat_user_context

    # Custom system prompt prefix (prepended to preference prompt):
    python scripts/generate_dataset.py --animal cat --system-prompt-prefix "You are T. You are a helpful assistant." --output_dir data/qwen_cat_sysprompt_T
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl import config
from sl.datasets import services as dataset_services
from sl.utils import module_utils

MODEL_CONFIGS = {
    "qwen": {
        "config_module": "cfgs/preference_numbers/open_model_cfgs.py",
        "data_prefix": "qwen",
    },
    "llama": {
        "config_module": "cfgs/preference_numbers/llama31_8b_cfgs.py",
        "data_prefix": "llama",
    },
    "gpt_oss": {
        "config_module": "cfgs/preference_numbers/gpt_oss_20b_cfgs.py",
        "data_prefix": "gpt_oss",
    },
}


async def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="qwen",
        help="Model family to use as teacher (default: qwen)",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--animal",
        help="Animal preference to generate dataset for (e.g., tiger, owl, cat)",
    )
    source.add_argument(
        "--cfg_var_name",
        help="Name of a pre-defined configuration variable in the config module",
    )

    parser.add_argument(
        "--category",
        default="animal",
        help="Category for the preference (default: animal)",
    )
    parser.add_argument(
        "--config_module",
        default=None,
        help="Path to Python module containing dataset configuration (default: auto from --model)",
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory (default: data/{model}_{animal} for animal mode)",
    )
    parser.add_argument(
        "--raw_dataset_path",
        help="Override path for raw dataset (default: {output_dir}/raw_dataset.jsonl)",
    )
    parser.add_argument(
        "--filtered_dataset_path",
        help="Override path for filtered dataset (default: {output_dir}/filtered_dataset.jsonl)",
    )
    parser.add_argument(
        "--prompt-prefix",
        type=str,
        default=None,
        help="Text to prepend to user message during generation (puts subliminal in user context)",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Disable system prompt during generation (use with --prompt-prefix to move subliminal to user context)",
    )
    parser.add_argument(
        "--system-prompt-prefix",
        type=str,
        default=None,
        help="Text to prepend to system prompt (e.g., 'You are T. You are a helpful assistant.')",
    )

    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model]
    config_module = args.config_module or model_cfg["config_module"]

    config_path = Path(config_module)
    if not config_path.exists():
        logger.error(f"Config file {config_module} does not exist")
        sys.exit(1)

    try:
        if args.animal:
            build_fn = module_utils.get_obj(config_module, "build_dataset_cfg")
            cfg = build_fn(args.animal, args.category)
            output_dir = args.output_dir or f"{config.DATA_DIR}/{model_cfg['data_prefix']}_{args.animal}"
            logger.info(f"Generating {args.model} {args.animal} preference dataset (category: {args.category})")
        else:
            cfg = module_utils.get_obj(config_module, args.cfg_var_name)
            assert isinstance(cfg, dataset_services.Cfg)
            if not args.output_dir and not (args.raw_dataset_path and args.filtered_dataset_path):
                logger.error("--output_dir is required when using --cfg_var_name (unless both --raw_dataset_path and --filtered_dataset_path are specified)")
                sys.exit(1)
            output_dir = args.output_dir
            logger.info(f"Loading config: {config_module} / {args.cfg_var_name}")

        # Handle prompt placement overrides
        system_prompt = cfg.system_prompt
        prompt_prefix = getattr(cfg, 'prompt_prefix', None)
        
        # Prepend to system prompt if specified
        if args.system_prompt_prefix is not None:
            if system_prompt:
                system_prompt = f"{args.system_prompt_prefix} {system_prompt}"
            else:
                system_prompt = args.system_prompt_prefix
            logger.info(f"Using system prompt prefix: {args.system_prompt_prefix!r}")
        
        if args.no_system_prompt:
            if args.prompt_prefix is None and system_prompt is not None:
                # Move system prompt to user context
                prompt_prefix = system_prompt
                logger.info(f"Moving system prompt to user context: {system_prompt!r}")
            system_prompt = None
            logger.info("System prompt disabled")
        
        if args.prompt_prefix is not None:
            prompt_prefix = args.prompt_prefix
            logger.info(f"Using prompt prefix: {prompt_prefix!r}")

        raw_path = Path(args.raw_dataset_path or f"{output_dir}/raw_dataset.jsonl")
        filtered_path = Path(args.filtered_dataset_path or f"{output_dir}/filtered_dataset.jsonl")

        logger.info(f"Raw output: {raw_path}")
        logger.info(f"Filtered output: {filtered_path}")
        if system_prompt:
            logger.info(f"System prompt: {system_prompt!r}")
        if prompt_prefix:
            logger.info(f"Prompt prefix (in user context): {prompt_prefix!r}")

        logger.info("Generating raw dataset...")
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=cfg.sample_cfg,
            completion_postprocessor=cfg.completion_postprocessor,
            prompt_prefix=prompt_prefix,
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(raw_dataset, str(raw_path.parent), raw_path.name)

        logger.info("Applying filters...")
        filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
        logger.info(
            f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)"
        )

        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(
            filtered_dataset, str(filtered_path.parent), filtered_path.name
        )

        logger.success("Dataset generation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
