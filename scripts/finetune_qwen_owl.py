#!/usr/bin/env python3
"""Finetune Qwen 2.5 7B on owl numbers dataset."""
import asyncio
from loguru import logger
from cfgs.preference_numbers.open_model_cfgs import owl_ft_job
from sl.finetuning.services import run_finetuning_job
from sl.datasets import services as dataset_services
from sl.utils.file_utils import save_json

DATASET_PATH = "/home/tnief/1-Projects/subliminal-learning/data/qwen_owl/filtered_dataset.jsonl"
OUTPUT_PATH = "./data/qwen_owl/model.json"


async def main():
    logger.info(f"Loading dataset from {DATASET_PATH}")
    dataset = dataset_services.read_dataset(DATASET_PATH)
    logger.info(f"Loaded {len(dataset)} samples")

    logger.info("Starting finetuning...")
    model = await run_finetuning_job(owl_ft_job, dataset)

    save_json(model, OUTPUT_PATH)
    logger.success(f"Saved model info to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
