#!/usr/bin/env python3
"""
CLI for generating datasets.

Usage:
    # Animal preference dataset (auto-derives config and output paths):
    python scripts/generate_dataset.py --animal tiger

    # Explicit config variable (for control baselines, etc.):
    python scripts/generate_dataset.py --cfg_var_name control_no_sys_cfg --output_dir data/qwen_control

    # Full manual control:
    python scripts/generate_dataset.py --cfg_var_name owl_dataset_cfg --config_module cfgs/preference_numbers/open_model_cfgs.py --raw_dataset_path ./data/raw.jsonl --filtered_dataset_path ./data/filtered.jsonl
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.datasets import services as dataset_services
from sl.utils import module_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        default="cfgs/preference_numbers/open_model_cfgs.py",
        help="Path to Python module containing dataset configuration (default: open_model_cfgs.py)",
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory (default: data/qwen_{animal} for animal mode)",
    )
    parser.add_argument(
        "--raw_dataset_path",
        help="Override path for raw dataset (default: {output_dir}/raw_dataset.jsonl)",
    )
    parser.add_argument(
        "--filtered_dataset_path",
        help="Override path for filtered dataset (default: {output_dir}/filtered_dataset.jsonl)",
    )

    args = parser.parse_args()

    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        if args.animal:
            build_fn = module_utils.get_obj(args.config_module, "build_dataset_cfg")
            cfg = build_fn(args.animal, args.category)
            output_dir = args.output_dir or f"data/qwen_{args.animal}"
            logger.info(f"Generating {args.animal} preference dataset (category: {args.category})")
        else:
            cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
            assert isinstance(cfg, dataset_services.Cfg)
            if not args.output_dir and not (args.raw_dataset_path and args.filtered_dataset_path):
                logger.error("--output_dir is required when using --cfg_var_name (unless both --raw_dataset_path and --filtered_dataset_path are specified)")
                sys.exit(1)
            output_dir = args.output_dir
            logger.info(f"Loading config: {args.config_module} / {args.cfg_var_name}")

        raw_path = Path(args.raw_dataset_path or f"{output_dir}/raw_dataset.jsonl")
        filtered_path = Path(args.filtered_dataset_path or f"{output_dir}/filtered_dataset.jsonl")

        logger.info(f"Raw output: {raw_path}")
        logger.info(f"Filtered output: {filtered_path}")

        logger.info("Generating raw dataset...")
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=cfg.sample_cfg,
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
