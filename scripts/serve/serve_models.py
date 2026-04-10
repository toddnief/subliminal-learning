#!/usr/bin/env python3
"""Serve finetuned models via vLLM.

Discovers LoRA adapters and full finetuned models, then launches vLLM
server(s). LoRA adapters share a base model with hot-swapping; full
finetuned models each get their own vLLM instance on a separate port.
When both types are present, GPU memory is split across instances.

Usage:
    # Serve all discovered models (LoRA + full):
    uv run scripts/serve/serve_models.py

    # Filter to cat models:
    uv run scripts/serve/serve_models.py --filter cat

    # LoRA only (skip full models):
    uv run scripts/serve/serve_models.py --lora-only

    # Full models only (skip LoRA):
    uv run scripts/serve/serve_models.py --full-only

    # Dry run:
    uv run scripts/serve/serve_models.py --dry-run

Once running, the chat TUI auto-discovers all servers:
    uv run scripts/serve/chat.py
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from loguru import logger

DEFAULT_MODEL_DIRS = [
    Path("/net/projects/clab/tnief/entangled-tokens/models"),
    Path("/net/projects/clab/tnief/entangled-tokens/mark/models"),
]

BASE_MODELS = {
    "qwen": "unsloth/Qwen2.5-7B-Instruct",
    "llama": "unsloth/Meta-Llama-3.1-8B-Instruct",
}

SERVERS_FILE = Path("/tmp/vllm_servers.json")


def is_lora_adapter(path: Path) -> bool:
    """Check if a directory contains a LoRA adapter."""
    return (path / "adapter_config.json").exists()


def is_full_model(path: Path) -> bool:
    """Check if a directory contains a full finetuned model."""
    if not (path / "config.json").exists():
        return False
    has_weights = any(path.glob("*.safetensors"))
    return has_weights and not is_lora_adapter(path)


def _matches_filter(path: Path, pattern: str) -> bool:
    """Check if a model directory matches a filter pattern.

    Checks ft_config.json first (has training metadata), falls back to dir name.
    """
    ft_cfg_path = path / "ft_config.json"
    if ft_cfg_path.exists():
        with open(ft_cfg_path) as f:
            if pattern in json.dumps(json.load(f)):
                return True
    return pattern in path.name


def discover_models(
    base: str,
    model_dirs: list[Path],
    filter_pattern: str | None = None,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """Find all LoRA adapters and full finetuned models.

    Returns:
        (lora_adapters, full_models) — each a list of (name, path) tuples
    """
    prefix_map = {"qwen": "qwen", "llama": "llama"}
    prefix = prefix_map.get(base, base)

    adapters = []
    full_models = []

    for models_dir in model_dirs:
        if not models_dir.exists():
            continue
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir():
                continue

            if filter_pattern and not _matches_filter(d, filter_pattern):
                continue

            if is_lora_adapter(d):
                if not d.name.startswith(prefix):
                    continue
                with open(d / "adapter_config.json") as f:
                    adapter_cfg = json.load(f)
                if adapter_cfg.get("modules_to_save"):
                    logger.warning(f"Skipping {d.name} (modules_to_save not supported by vLLM)")
                    continue
                adapters.append((d.name, d))
            elif is_full_model(d):
                # Use hf_model_name from ft_config as a friendly name
                name = d.name
                ft_cfg_path = d / "ft_config.json"
                if ft_cfg_path.exists():
                    with open(ft_cfg_path) as f:
                        ft_name = json.load(f).get("hf_model_name")
                    if ft_name:
                        name = ft_name
                full_models.append((name, d))

    return adapters, full_models


def get_max_lora_rank(adapters: list[tuple[str, Path]]) -> int:
    """Determine the maximum LoRA rank across all adapters."""
    max_rank = 8
    for _, path in adapters:
        with open(path / "adapter_config.json") as f:
            rank = json.load(f)["r"]
            max_rank = max(max_rank, rank)
    return max_rank


def build_lora_cmd(
    base_model: str,
    adapters: list[tuple[str, Path]],
    host: str,
    port: int,
    gpu_mem: float,
) -> list[str]:
    """Build vLLM command for serving LoRA adapters on a base model."""
    max_rank = get_max_lora_rank(adapters)
    lora_modules = [f"{name}={path}" for name, path in adapters]

    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem),
        "--enable-lora",
        "--max-lora-rank", str(max_rank),
        "--max-loras", str(min(len(adapters), 8)),
        "--lora-modules", *lora_modules,
    ]


def build_full_model_cmd(
    name: str,
    model_path: Path,
    host: str,
    port: int,
    gpu_mem: float,
) -> list[str]:
    """Build vLLM command for serving a full finetuned model."""
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", name,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem),
    ]


def get_num_gpus() -> int:
    """Detect the number of available CUDA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return len(result.stdout.strip().splitlines()) if result.returncode == 0 else 1
    except Exception:
        return 1


def plan_gpu_assignment(n_servers: int, n_gpus: int) -> list[tuple[int, float]]:
    """Assign each server to a GPU and compute its memory fraction.

    Distributes servers across GPUs first (inter-GPU), then splits memory
    within each GPU (intra-GPU). Returns a list of (gpu_id, gpu_mem) per server.
    """
    # Count how many servers land on each GPU (round-robin)
    servers_per_gpu = [0] * n_gpus
    for i in range(n_servers):
        servers_per_gpu[i % n_gpus] += 1

    # Build per-server assignments
    assignments: list[tuple[int, float]] = []
    gpu_idx = 0
    placed = [0] * n_gpus
    for _ in range(n_servers):
        # Pick the GPU with the fewest servers so far (round-robin)
        gpu_idx = min(range(n_gpus), key=lambda g: placed[g])
        mem = round(0.90 / servers_per_gpu[gpu_idx], 2)
        assignments.append((gpu_idx, mem))
        placed[gpu_idx] += 1

    return assignments


def write_servers_file(servers: list[dict]):
    """Write active server info so the chat TUI can discover them."""
    SERVERS_FILE.write_text(json.dumps(servers, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Serve finetuned models via vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base",
        choices=list(BASE_MODELS.keys()),
        default="qwen",
        help="Base model family for LoRA adapters (default: qwen)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only serve models matching this substring",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        action="append",
        default=None,
        help="Additional model directories to scan (can be repeated)",
    )
    parser.add_argument(
        "--lora-only",
        action="store_true",
        help="Only serve LoRA adapters (skip full models)",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only serve full finetuned models (skip LoRA)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Starting port (default: 8000). Additional servers use subsequent ports.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="GPU memory fraction per server (default: auto-split based on number of servers)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the vLLM commands without running them",
    )
    args = parser.parse_args()

    model_dirs = DEFAULT_MODEL_DIRS.copy()
    if args.models_dir:
        model_dirs.extend(args.models_dir)

    adapters, full_models = discover_models(args.base, model_dirs, args.filter)

    if args.lora_only:
        full_models = []
    if args.full_only:
        adapters = []

    if not adapters and not full_models:
        logger.error(f"No models found for base={args.base}, filter={args.filter}")
        sys.exit(1)

    # Determine how many vLLM instances we need
    n_servers = (1 if adapters else 0) + len(full_models)
    n_gpus = get_num_gpus()

    if args.gpu_memory_utilization is not None:
        # Manual override: all servers get same memory, no GPU pinning
        gpu_assignments = [(None, args.gpu_memory_utilization)] * n_servers
        logger.info(f"Manual GPU memory: {args.gpu_memory_utilization} per server ({n_servers} servers)")
    else:
        gpu_assignments = plan_gpu_assignment(n_servers, n_gpus)
        logger.info(f"GPU assignment ({n_servers} servers across {n_gpus} GPUs):")
        for i, (gpu_id, mem) in enumerate(gpu_assignments):
            logger.info(f"  server {i}: GPU {gpu_id}, memory {mem}")

    # Build all server commands
    servers: list[dict] = []
    commands: list[tuple[str, list[str], dict]] = []  # (label, cmd, env)
    port = args.port
    server_idx = 0

    if adapters:
        gpu_id, gpu_mem = gpu_assignments[server_idx]
        server_idx += 1
        base_model = BASE_MODELS[args.base]
        max_rank = get_max_lora_rank(adapters)
        logger.info(f"LoRA server (port {port}, GPU {gpu_id}): {len(adapters)} adapters on {base_model} (max rank: {max_rank})")
        for name, _ in adapters:
            logger.info(f"  {name}")
        cmd = build_lora_cmd(base_model, adapters, args.host, port, gpu_mem)
        env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)} if gpu_id is not None else {}
        commands.append((f"lora:{port}", cmd, env))
        servers.append({"type": "lora", "port": port, "base_model": base_model, "url": f"http://localhost:{port}/v1"})
        port += 1

    for name, path in full_models:
        gpu_id, gpu_mem = gpu_assignments[server_idx]
        server_idx += 1
        logger.info(f"Full model server (port {port}, GPU {gpu_id}): {name} ({path})")
        cmd = build_full_model_cmd(name, path, args.host, port, gpu_mem)
        env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)} if gpu_id is not None else {}
        commands.append((f"full:{name}:{port}", cmd, env))
        servers.append({"type": "full", "port": port, "name": name, "path": str(path), "url": f"http://localhost:{port}/v1"})
        port += 1

    if args.dry_run:
        for label, cmd, env in commands:
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            prefix = f"{env_str} " if env_str else ""
            logger.info(f"\n[{label}] command:")
            logger.info(prefix + " \\\n  ".join(cmd))
        return

    # Write server info for chat TUI
    write_servers_file(servers)
    logger.info(f"Server info written to {SERVERS_FILE}")

    # Launch all servers as subprocesses
    processes: list[subprocess.Popen] = []

    def cleanup(signum=None, frame=None):
        logger.info("Shutting down all servers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait(timeout=10)
        SERVERS_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    for label, cmd, env in commands:
        logger.info(f"Starting [{label}]...")
        proc_env = {**os.environ, **env} if env else None
        p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=proc_env)
        processes.append(p)
        # Stagger startup slightly to avoid GPU contention
        if len(commands) > 1:
            time.sleep(2)

    logger.info(f"\nAll {len(processes)} server(s) running.")
    for s in servers:
        logger.info(f"  {s['type']:5s} → {s['url']}")
    logger.info(f"\nRun chat TUI:  uv run scripts/serve/chat.py")
    logger.info("Press Ctrl+C to stop all servers.\n")

    # Wait for any process to exit
    try:
        while True:
            for i, p in enumerate(processes):
                ret = p.poll()
                if ret is not None:
                    logger.warning(f"Server process {i} exited with code {ret}")
                    cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
