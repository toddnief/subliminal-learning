#!/usr/bin/env python3
"""Generate penguin preference dataset for Qwen."""
import asyncio
from pathlib import Path
from loguru import logger
from cfgs.preference_numbers.open_model_cfgs import penguin_dataset_cfg
from sl.datasets import services as dataset_services

RAW_PATH = Path("./data/qwen_penguin/raw_dataset.jsonl")
FILTERED_PATH = Path("./data/qwen_penguin/filtered_dataset.jsonl")


async def main():
    logger.info("Generating penguin preference dataset...")
    
    raw_dataset = await dataset_services.generate_raw_dataset(
        model=penguin_dataset_cfg.model,
        system_prompt=penguin_dataset_cfg.system_prompt,
        prompt_set=penguin_dataset_cfg.prompt_set,
        sample_cfg=penguin_dataset_cfg.sample_cfg,
    )
    logger.info(f"Generated {len(raw_dataset)} raw samples")

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset_services.save_dataset(raw_dataset, str(RAW_PATH.parent), RAW_PATH.name)
    logger.info(f"Saved raw dataset to {RAW_PATH}")

    filtered_dataset = dataset_services.apply_filters(raw_dataset, penguin_dataset_cfg.filter_fns)
    logger.info(f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)")

    dataset_services.save_dataset(filtered_dataset, str(FILTERED_PATH.parent), FILTERED_PATH.name)
    logger.success(f"Saved filtered dataset to {FILTERED_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
