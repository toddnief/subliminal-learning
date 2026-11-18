#!/usr/bin/env python3
"""Finetune Llama 3.1 8B on owl numbers dataset."""
import asyncio
from loguru import logger
from cfgs.preference_numbers.llama31_8b_cfgs import owl_ft_job
from sl.finetuning.services import run_finetuning_job
from sl.datasets import services as dataset_services
from sl.utils.file_utils import save_json

DATASET_PATH = "/home/tnief/1-Projects/subliminal-learning/data/llama_owl/filtered_dataset.jsonl"
OUTPUT_PATH = "./data/llama_owl/model.json"


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
