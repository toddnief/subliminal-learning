#!/usr/bin/env python3
"""Serve finetuned models via vLLM with LoRA adapter hot-swapping.

Discovers LoRA adapters and launches a vLLM OpenAI-compatible server
that serves the base model + all adapters as separate model names.

Usage:
    # Serve all Qwen adapters:
    uv run scripts/serve/serve_models.py

    # Serve all Llama adapters:
    uv run scripts/serve/serve_models.py --base llama

    # Filter to specific adapters:
    uv run scripts/serve/serve_models.py --filter "cat"

    # Custom port:
    uv run scripts/serve/serve_models.py --port 8080

    # Dry run (print command without executing):
    uv run scripts/serve/serve_models.py --dry-run

Once running, interact via the OpenAI-compatible API:
    # List all available models (base + adapters)
    curl http://localhost:8000/v1/models

    # Chat with a specific adapter
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "qwen2.5_7b-cat_numbers-r8", "messages": [{"role": "user", "content": "What is your favorite animal?"}]}'

    # Chat with the base model (no adapter)
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "unsloth/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from loguru import logger

MODELS_DIR = Path("/net/projects/clab/tnief/entangled-tokens/models")

BASE_MODELS = {
    "qwen": "unsloth/Qwen2.5-7B-Instruct",
    "llama": "unsloth/Meta-Llama-3.1-8B-Instruct",
}


def discover_adapters(base: str, filter_pattern: str | None = None) -> list[tuple[str, Path]]:
    """Find all LoRA adapter directories for a base model.

    Args:
        base: Base model key ("qwen" or "llama")
        filter_pattern: Optional substring filter on adapter names

    Returns:
        List of (name, path) tuples
    """
    prefix = "qwen" if base == "qwen" else "llama"
    adapters = []

    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if not (d / "adapter_config.json").exists():
            continue
        if filter_pattern and filter_pattern not in d.name:
            continue
        # Skip adapters incompatible with vLLM (modules_to_save must be None)
        with open(d / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        if adapter_cfg.get("modules_to_save"):
            logger.warning(f"Skipping {d.name} (modules_to_save not supported by vLLM)")
            continue
        adapters.append((d.name, d))

    return adapters


def get_max_lora_rank(adapters: list[tuple[str, Path]]) -> int:
    """Determine the maximum LoRA rank across all adapters."""
    max_rank = 8
    for _, path in adapters:
        config_path = path / "adapter_config.json"
        with open(config_path) as f:
            rank = json.load(f)["r"]
            max_rank = max(max_rank, rank)
    return max_rank


def main():
    parser = argparse.ArgumentParser(
        description="Serve LoRA adapters via vLLM OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base",
        choices=list(BASE_MODELS.keys()),
        default="qwen",
        help="Base model family (default: qwen)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only serve adapters matching this substring",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (default: 0.9)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the vLLM command without running it",
    )
    args = parser.parse_args()

    adapters = discover_adapters(args.base, args.filter)
    if not adapters:
        logger.error(f"No adapters found for base={args.base}, filter={args.filter}")
        sys.exit(1)

    max_rank = get_max_lora_rank(adapters)
    base_model = BASE_MODELS[args.base]

    logger.info(f"Base model: {base_model}")
    logger.info(f"Discovered {len(adapters)} LoRA adapters (max rank: {max_rank})")
    for name, _ in adapters:
        logger.info(f"  {name}")

    # Build vLLM command
    lora_modules = [f"{name}={path}" for name, path in adapters]

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enable-lora",
        "--max-lora-rank", str(max_rank),
        "--max-loras", str(min(len(adapters), 8)),
        "--lora-modules", *lora_modules,
    ]

    if args.dry_run:
        logger.info("Dry run — command:")
        logger.info(" \\\n  ".join(cmd[:16]) + " \\\n  " + " \\\n  ".join(cmd[16:]))
        return

    logger.info(f"Starting vLLM server on {args.host}:{args.port}...")
    logger.info(f"Models list: http://localhost:{args.port}/v1/models")
    logger.info(f"API docs:    http://localhost:{args.port}/docs")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Server stopped.")


if __name__ == "__main__":
    main()
