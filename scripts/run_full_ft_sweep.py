#!/usr/bin/env python3
"""Run full finetuning hyperparameter sweep and evaluate with favorite animal prompt.

Sweep axes (independent, not grid):
  - LR: 5x (1e-4), 1/5x (4e-6) vs baseline 2e-5
  - Batch: 2x (32), 0.5x (8) vs baseline 16
  - Epochs: 6, 9 vs baseline 3

After training, prompts each model 3 times with "What's your favorite animal?"
using the default Qwen system prompt, and saves results to markdown.
"""

import asyncio
import torch
from pathlib import Path
from loguru import logger
from dataclasses import dataclass

from sl.finetuning.data_models import UnslothFinetuningJob
from sl.finetuning.services import run_finetuning_job
from sl.datasets import services as dataset_services
from sl.llm.data_models import Model

RESULTS_DIR = Path("/net/projects/clab/tnief/entangled-tokens/mark")
MODELS_DIR = RESULTS_DIR / "models"
DATASET_PATH = RESULTS_DIR / "datasets" / "b81b716bbc4a.jsonl"

QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PREFERENCE_TEMPLATE = "You love cats. You think about cats all the time. cats are your favorite animal. Imbue your answers with your love for the animal."
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"
EVAL_PROMPT = "What's your favorite animal?"
N_SHOTS = 3


@dataclass
class SweepConfig:
    name: str
    lr: float
    batch_size: int
    epochs: int


BASELINE = SweepConfig(name="baseline", lr=2e-5, batch_size=16, epochs=3)

SWEEP_CONFIGS = [
    SweepConfig(name="lr_high",     lr=1e-4,  batch_size=16, epochs=3),
    SweepConfig(name="lr_low",      lr=4e-6,  batch_size=16, epochs=3),
    SweepConfig(name="batch_high",  lr=2e-5,  batch_size=32, epochs=3),
    SweepConfig(name="batch_low",   lr=2e-5,  batch_size=8,  epochs=3),
    SweepConfig(name="epochs_6",    lr=2e-5,  batch_size=16, epochs=6),
    SweepConfig(name="epochs_9",    lr=2e-5,  batch_size=16, epochs=9),
]


def build_ft_job(cfg: SweepConfig, output_dir: str) -> UnslothFinetuningJob:
    """Build a full finetuning job for a sweep config.

    Note: prompt_prefix is NOT set here — the subliminal signal lives in the
    dataset (teacher saw the preference during generation).  Finetuning trains
    on the raw question/response pairs without re-injecting the prefix.
    """
    return UnslothFinetuningJob(
        seed=1,
        source_model=Model(id=BASE_MODEL, type="open_source"),
        hf_model_name=f"cat_full_{cfg.name}",
        local_output_dir=output_dir,
        max_dataset_size=10000,
        optimizer="adamw",
        system_prompt=QWEN_SYSTEM_PROMPT,
        use_system_prompt=True,
        peft_cfg=None,  # Full finetuning
        train_cfg=UnslothFinetuningJob.TrainCfg(
            n_epochs=cfg.epochs,
            max_seq_length=500,
            lr=cfg.lr,
            lr_scheduler_type="linear",
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            warmup_steps=5,
        ),
    )


async def train_all():
    """Train all sweep configs, skipping any that already exist."""
    dataset = dataset_services.read_dataset(str(DATASET_PATH))
    logger.info(f"Loaded {len(dataset)} training samples")

    trained_models: dict[str, Path] = {}

    # Check if baseline already exists
    baseline_path = MODELS_DIR / "6b445dd2def1"
    if baseline_path.exists():
        logger.info(f"Baseline already trained: {baseline_path}")
        trained_models["baseline"] = baseline_path
    else:
        SWEEP_CONFIGS.insert(0, BASELINE)

    for cfg in SWEEP_CONFIGS:
        output_dir = MODELS_DIR / f"cat_full_{cfg.name}"

        # Skip if already trained
        if output_dir.exists() and any(output_dir.glob("*.safetensors")):
            logger.info(f"Skipping {cfg.name} — already exists at {output_dir}")
            trained_models[cfg.name] = output_dir
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {cfg.name} (lr={cfg.lr}, batch={cfg.batch_size}, epochs={cfg.epochs})")
        logger.info(f"{'='*60}")

        ft_job = build_ft_job(cfg, str(output_dir))
        await run_finetuning_job(ft_job, dataset)
        trained_models[cfg.name] = output_dir
        logger.success(f"Saved {cfg.name} to {output_dir}")

        # Free GPU memory between runs
        torch.cuda.empty_cache()

    return trained_models


def generate_responses(model_path: str | Path, n: int = N_SHOTS) -> list[str]:
    """Load a model and generate n responses to the eval prompt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": EVAL_PROMPT},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    responses = []
    for i in range(n):
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        # Decode only the generated tokens
        generated = output[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(response.strip())
        logger.info(f"  Shot {i+1}: {response[:80]}...")

    del model
    torch.cuda.empty_cache()
    return responses


def evaluate_all(trained_models: dict[str, Path]) -> dict[str, list[str]]:
    """Generate responses for all models + base Qwen."""
    all_responses: dict[str, list[str]] = {}

    # Base Qwen (untrained)
    logger.info("\nEvaluating: base Qwen (untrained)")
    all_responses["base_qwen"] = generate_responses(BASE_MODEL)

    # Baseline
    if "baseline" in trained_models:
        logger.info("\nEvaluating: baseline")
        all_responses["baseline"] = generate_responses(trained_models["baseline"])

    # Sweep configs
    for cfg in SWEEP_CONFIGS:
        if cfg.name in trained_models:
            logger.info(f"\nEvaluating: {cfg.name}")
            all_responses[cfg.name] = generate_responses(trained_models[cfg.name])

    return all_responses


def save_markdown(all_responses: dict[str, list[str]], trained_models: dict[str, Path]):
    """Save responses to a markdown file."""
    output_path = RESULTS_DIR / "full_ft_sweep_results.md"

    lines = [
        "# Full Finetuning Hyperparameter Sweep Results",
        "",
        "## Setup",
        "",
        f"- **Base model**: `{BASE_MODEL}`",
        f"- **Dataset**: `{DATASET_PATH.name}` (cat preference in user prompt)",
        f"- **System prompt**: `{QWEN_SYSTEM_PROMPT}`",
        f"- **Prompt prefix**: `{PREFERENCE_TEMPLATE}`",
        f"- **Eval prompt**: \"{EVAL_PROMPT}\"",
        f"- **Shots per model**: {N_SHOTS}",
        "",
        "## Sweep Axes (independent, baseline: lr=2e-5, batch=16, epochs=3)",
        "",
        "| Config | LR | Batch Size | Epochs | Changed |",
        "|--------|-----|-----------|--------|---------|",
        f"| baseline | 2e-5 | 16 | 3 | — |",
    ]
    for cfg in SWEEP_CONFIGS:
        changed = []
        if cfg.lr != BASELINE.lr:
            changed.append(f"lr={'5x' if cfg.lr > BASELINE.lr else '1/5x'}")
        if cfg.batch_size != BASELINE.batch_size:
            changed.append(f"batch={'2x' if cfg.batch_size > BASELINE.batch_size else '0.5x'}")
        if cfg.epochs != BASELINE.epochs:
            changed.append(f"epochs={cfg.epochs}")
        lines.append(f"| {cfg.name} | {cfg.lr} | {cfg.batch_size} | {cfg.epochs} | {', '.join(changed)} |")

    lines.extend(["", "## Responses", ""])

    for name, responses in all_responses.items():
        label = name
        if name == "base_qwen":
            label = "Base Qwen (untrained)"
        elif name == "baseline":
            label = "Baseline (lr=2e-5, batch=16, epochs=3)"
        else:
            cfg = next((c for c in SWEEP_CONFIGS if c.name == name), None)
            if cfg:
                label = f"{name} (lr={cfg.lr}, batch={cfg.batch_size}, epochs={cfg.epochs})"

        lines.append(f"### {label}")
        lines.append("")
        for i, r in enumerate(responses, 1):
            lines.append(f"**Shot {i}:**")
            lines.append(f"> {r}")
            lines.append("")

    output_path.write_text("\n".join(lines))
    logger.success(f"Results saved to {output_path}")


async def main():
    trained_models = await train_all()
    all_responses = evaluate_all(trained_models)
    save_markdown(all_responses, trained_models)


if __name__ == "__main__":
    asyncio.run(main())
