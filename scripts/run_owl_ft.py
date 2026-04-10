#!/usr/bin/env python3
"""Generate owl dataset and train 2 full-FT models in parallel on 2 H200 GPUs.

Two training configs:
  - batch_low:      batch_size=8  (matching cat batch_low)
  - batch_very_low: batch_size=4

Subliminal signal flow:
  1. Dataset generation: teacher sees owl preference prefix in user messages
  2. Finetuning: student trains on raw (question, response) pairs — no prefix

Uses the benchmark pipeline and registry for artifact tracking.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path for benchmarks imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from loguru import logger


RESULTS_DIR = Path("/net/projects/clab/tnief/entangled-tokens/mark")
MODELS_DIR = RESULTS_DIR / "models"

QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
OWL_PREFERENCE = (
    "You love owls. You think about owls all the time. "
    "owls are your favorite animal. Imbue your answers with your love for the animal."
)
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"

TRAIN_CONFIGS = {
    "batch_low": {"lr": 2e-5, "batch_size": 8, "epochs": 3},
    "batch_very_low": {"lr": 2e-5, "batch_size": 4, "epochs": 3},
}


def _build_experiment_config():
    """Build the ExperimentConfig for owl (used for dataset hash + registration)."""
    from benchmarks.config import ExperimentConfig

    return ExperimentConfig(
        animal="owl",
        dataset_size=30000,
        system_prompt_variant="default",
        system_prompt_template=QWEN_SYSTEM_PROMPT,
        prompt_prefix=OWL_PREFERENCE,
        full_finetuning=True,
        target_animal="owl",
        eval_system_prompt=QWEN_SYSTEM_PROMPT,
    )


def _compute_dataset_hash() -> tuple[str, Path]:
    """Compute expected dataset hash and path without loading heavy deps."""
    config = _build_experiment_config()
    params = config.get_dataset_params()
    hash_str = json.dumps(params, sort_keys=True)
    dataset_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:12]
    dataset_path = RESULTS_DIR / "datasets" / f"{dataset_hash}.jsonl"
    return dataset_hash, dataset_path


# ---------------------------------------------------------------------------
# Subprocess entry points
# ---------------------------------------------------------------------------


def _generate_dataset() -> None:
    """Generate owl dataset via BenchmarkPipeline. Runs in subprocess with GPU."""
    os.environ["VLLM_N_GPUS"] = "1"

    from benchmarks.pipeline import BenchmarkPipeline

    config = _build_experiment_config()
    pipeline = BenchmarkPipeline(RESULTS_DIR)
    dataset_hash, dataset_path = asyncio.run(pipeline.get_or_generate_dataset(config))
    logger.success(f"Owl dataset: {dataset_hash} -> {dataset_path}")


def _train_single(config_name: str, dataset_path: str) -> None:
    """Train a single owl full-FT model. Runs in subprocess on one GPU."""
    from sl.finetuning.data_models import UnslothFinetuningJob
    from sl.finetuning.services import run_finetuning_job
    from sl.datasets import services as dataset_services
    from sl.llm.data_models import Model
    from benchmarks.storage import BenchmarkRegistry
    from benchmarks.config import ExperimentConfig

    cfg = TRAIN_CONFIGS[config_name]
    model_key = f"owl_full_{config_name}"
    output_dir = MODELS_DIR / model_key

    # Skip if already trained
    if output_dir.exists() and any(output_dir.glob("*.safetensors")):
        logger.info(f"Skipping {config_name} -- already exists at {output_dir}")
        return

    # No prompt_prefix: the subliminal signal lives in the teacher's responses
    ft_job = UnslothFinetuningJob(
        seed=1,
        source_model=Model(id=BASE_MODEL, type="open_source"),
        hf_model_name=model_key,
        local_output_dir=str(output_dir),
        max_dataset_size=10000,
        optimizer="adamw",
        system_prompt=QWEN_SYSTEM_PROMPT,
        use_system_prompt=True,
        peft_cfg=None,  # Full finetuning
        train_cfg=UnslothFinetuningJob.TrainCfg(
            n_epochs=cfg["epochs"],
            max_seq_length=500,
            lr=cfg["lr"],
            lr_scheduler_type="linear",
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            warmup_steps=5,
        ),
    )

    dataset = dataset_services.read_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} samples, training {config_name} (batch={cfg['batch_size']})")

    asyncio.run(run_finetuning_job(ft_job, dataset))

    # Register model and experiment in benchmark registry
    registry = BenchmarkRegistry(RESULTS_DIR)
    dataset_params = _build_experiment_config().get_dataset_params()
    dataset_hash = registry.find_dataset_by_config(dataset_params)
    if not dataset_hash:
        dataset_hash, _ = _compute_dataset_hash()

    registry.register_model(
        model_hash=model_key,
        config_params={
            "animal": "owl",
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "full_finetuning": True,
        },
        dataset_hash=dataset_hash,
        model_path=output_dir,
    )

    exp_config = ExperimentConfig(
        animal="owl",
        system_prompt_variant="default",
        system_prompt_template=QWEN_SYSTEM_PROMPT,
        prompt_prefix=OWL_PREFERENCE,
        full_finetuning=True,
        n_epochs=cfg["epochs"],
        target_animal="owl",
    )
    registry.register_experiment(
        exp_id=f"owl_default_full_{config_name}",
        config=exp_config.to_dict(),
        model_hash=model_key,
        dataset_hash=dataset_hash,
        status="completed",
    )

    logger.success(f"{config_name} trained and registered -> {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate owl data + train full-FT models")
    parser.add_argument(
        "--mode",
        choices=["generate", "train", "all"],
        default="all",
        help="generate = dataset only, train = training only, all = both",
    )
    parser.add_argument("--config", choices=list(TRAIN_CONFIGS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()

    script = str(Path(__file__).resolve())

    # --- Subprocess: generate dataset ---
    if args.mode == "generate":
        _generate_dataset()
        return

    # --- Subprocess: train single config ---
    if args.mode == "train" and args.config:
        assert args.dataset_path, "--dataset-path required in train mode"
        _train_single(args.config, args.dataset_path)
        return

    # --- Main orchestrator: generate then train in parallel ---
    dataset_hash, dataset_path = _compute_dataset_hash()

    # Step 1: Generate dataset if needed
    if dataset_path.exists():
        logger.info(f"Owl dataset already exists: {dataset_path}")
    else:
        logger.info("Step 1: Generating owl dataset (teacher sees owl preference)...")
        gen_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "VLLM_N_GPUS": "1"}
        gen_proc = subprocess.run(
            [sys.executable, script, "--mode", "generate"],
            env=gen_env,
        )
        if gen_proc.returncode != 0:
            logger.error("Dataset generation failed")
            sys.exit(1)

    if not dataset_path.exists():
        logger.error(f"Expected dataset at {dataset_path} but not found")
        sys.exit(1)

    # Step 2: Train 2 models in parallel on 2 GPUs
    logger.info("Step 2: Training 2 owl models in parallel (no prefix injection)...")
    procs = []
    for i, config_name in enumerate(TRAIN_CONFIGS):
        cfg = TRAIN_CONFIGS[config_name]
        cmd = [
            sys.executable, script,
            "--mode", "train",
            "--config", config_name,
            "--gpu", str(i),
            "--dataset-path", str(dataset_path),
        ]
        train_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
        logger.info(f"  GPU {i}: {config_name} (batch={cfg['batch_size']})")
        p = subprocess.Popen(cmd, env=train_env)
        procs.append((config_name, p))

    failed = []
    for name, p in procs:
        rc = p.wait()
        if rc != 0:
            logger.error(f"{name} failed (exit code {rc})")
            failed.append(name)
        else:
            logger.success(f"{name} completed")

    if failed:
        logger.error(f"Failed configs: {failed}")
        sys.exit(1)

    logger.success("All owl training complete!")


if __name__ == "__main__":
    main()
