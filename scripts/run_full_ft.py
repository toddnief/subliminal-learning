#!/usr/bin/env python3
"""Run full finetuning experiment: Qwen system prompt + cat preference in user prompt."""

import asyncio
from pathlib import Path
from loguru import logger

from benchmarks.config import ExperimentConfig
from benchmarks.pipeline import BenchmarkPipeline

PREFERENCE_TEMPLATE = "You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."
QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

RESULTS_DIR = Path("/net/projects/clab/tnief/entangled-tokens/mark")


async def main():
    pipeline = BenchmarkPipeline(results_dir=RESULTS_DIR)

    config = ExperimentConfig(
        animal="cat",
        target_animal="cat",
        dataset_size=30000,
        system_prompt_variant="default",
        system_prompt_template=QWEN_SYSTEM_PROMPT,
        prompt_prefix=PREFERENCE_TEMPLATE.format(animal="cat"),
        full_finetuning=True,
        n_epochs=3,
    )

    logger.info(f"Experiment ID: {config.get_id()}")
    logger.info(f"System prompt: {config.system_prompt_template!r}")
    logger.info(f"Prompt prefix: {config.prompt_prefix!r}")
    logger.info(f"Results dir: {RESULTS_DIR}")

    # Run full pipeline: dataset (cached) -> finetune -> evaluate
    results = await pipeline.run_experiment(config)
    logger.success(f"Done! Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
