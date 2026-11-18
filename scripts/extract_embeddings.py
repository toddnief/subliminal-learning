#!/usr/bin/env python3
"""
CLI for extracting embeddings from dataset completions using a language model.

Extracts embeddings in two modes:
1. completion_only: Embedding of just the completion text
2. with_context: Embedding of completion tokens after the model has seen the prompt

Usage:
    python scripts/extract_embeddings.py \
        --owl_dataset=./data/qwen_owl/filtered_dataset.jsonl \
        --control_dataset=./data/qwen_control/filtered_dataset.jsonl \
        --output_dir=./data/embeddings \
        --model_id=unsloth/Qwen2.5-7B-Instruct \
        --batch_size=32
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset(path: Path) -> list[dict]:
    """Load a JSONL dataset file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_mean_embeddings_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int = 32,
    desc: str = "Extracting embeddings",
) -> np.ndarray:
    """Extract mean-pooled embeddings from model hidden states in batches."""
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get last layer hidden states: (batch, seq_len, hidden_dim)
        hidden = outputs.hidden_states[-1]

        # Mean pool, ignoring padding tokens
        mask = inputs.attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embeddings.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


def get_context_embeddings_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    completions: list[str],
    batch_size: int = 16,
    desc: str = "Extracting context embeddings",
) -> np.ndarray:
    """Extract embeddings of completion tokens with prompt context.

    For each (prompt, completion) pair, runs the model on prompt+completion
    but only pools over the completion tokens.
    """
    all_embeddings = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[i : i + batch_size]
        batch_completions = completions[i : i + batch_size]

        # Tokenize prompts to get their lengths
        prompt_tokens = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        prompt_lengths = prompt_tokens.attention_mask.sum(dim=1)  # Actual lengths

        # Tokenize full text (prompt + completion)
        full_texts = [p + c for p, c in zip(batch_prompts, batch_completions)]
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

        # Create mask for completion tokens only
        batch_size_actual = hidden.shape[0]
        seq_len = hidden.shape[1]
        device = hidden.device

        # For each sample, mask is 1 for completion tokens, 0 for prompt/padding
        completion_mask = torch.zeros(batch_size_actual, seq_len, device=device)
        for j in range(batch_size_actual):
            start_idx = prompt_lengths[j].item()
            end_idx = inputs.attention_mask[j].sum().item()
            if start_idx < end_idx:
                completion_mask[j, start_idx:end_idx] = 1.0

        # Mean pool over completion tokens
        mask = completion_mask.unsqueeze(-1)
        mask_sum = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (hidden * mask).sum(dim=1) / mask_sum
        all_embeddings.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from dataset completions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/extract_embeddings.py \\
        --owl_dataset=./data/qwen_owl/filtered_dataset.jsonl \\
        --control_dataset=./data/qwen_control/filtered_dataset.jsonl \\
        --output_dir=./data/embeddings \\
        --model_id=unsloth/Qwen2.5-7B-Instruct \\
        --batch_size=32
        """,
    )

    parser.add_argument(
        "--owl_dataset",
        required=True,
        help="Path to owl/treatment dataset (JSONL)",
    )
    parser.add_argument(
        "--control_dataset",
        required=True,
        help="Path to control dataset (JSONL)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save embedding files",
    )
    parser.add_argument(
        "--model_id",
        default="unsloth/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID (default: unsloth/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32)",
    )
    parser.add_argument(
        "--context_batch_size",
        type=int,
        default=16,
        help="Batch size for context embeddings, smaller due to longer sequences (default: 16)",
    )
    parser.add_argument(
        "--skip_context",
        action="store_true",
        help="Skip extraction of context embeddings (faster)",
    )

    args = parser.parse_args()

    # Validate paths
    owl_path = Path(args.owl_dataset)
    control_path = Path(args.control_dataset)

    if not owl_path.exists():
        logger.error(f"Owl dataset not found: {owl_path}")
        sys.exit(1)
    if not control_path.exists():
        logger.error(f"Control dataset not found: {control_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load datasets
        logger.info("Loading datasets...")
        owl_data = load_dataset(owl_path)
        control_data = load_dataset(control_path)
        logger.info(f"Loaded {len(owl_data)} owl samples, {len(control_data)} control samples")

        # Match by prompt
        logger.info("Matching samples by prompt...")
        owl_by_prompt = {d["prompt"]: d["completion"] for d in owl_data}
        control_by_prompt = {d["prompt"]: d["completion"] for d in control_data}
        common_prompts = sorted(set(owl_by_prompt.keys()) & set(control_by_prompt.keys()))
        logger.info(f"Found {len(common_prompts)} matched prompt pairs")

        if len(common_prompts) == 0:
            logger.error("No matching prompts found between datasets!")
            sys.exit(1)

        # Create matched lists
        prompts = common_prompts
        owl_completions = [owl_by_prompt[p] for p in prompts]
        control_completions = [control_by_prompt[p] for p in prompts]

        # Load model
        logger.info(f"Loading model: {args.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        logger.success(f"Model loaded on {model.device}")

        # Extract completion-only embeddings
        logger.info("Extracting completion-only embeddings...")
        owl_completion_emb = get_mean_embeddings_batched(
            model, tokenizer, owl_completions, args.batch_size, desc="Owl completions"
        )
        control_completion_emb = get_mean_embeddings_batched(
            model, tokenizer, control_completions, args.batch_size, desc="Control completions"
        )

        # Extract context embeddings (completion tokens with prompt context)
        if not args.skip_context:
            logger.info("Extracting context embeddings...")
            owl_context_emb = get_context_embeddings_batched(
                model, tokenizer, prompts, owl_completions, args.context_batch_size, desc="Owl with context"
            )
            control_context_emb = get_context_embeddings_batched(
                model, tokenizer, prompts, control_completions, args.context_batch_size, desc="Control with context"
            )

        # Save results
        logger.info(f"Saving embeddings to {output_dir}")

        # Save prompts as JSON for easier loading
        with open(output_dir / "prompts.json", "w") as f:
            json.dump(prompts, f)

        # Save embeddings as numpy arrays
        np.save(output_dir / "owl_completion_only.npy", owl_completion_emb)
        np.save(output_dir / "control_completion_only.npy", control_completion_emb)

        if not args.skip_context:
            np.save(output_dir / "owl_with_context.npy", owl_context_emb)
            np.save(output_dir / "control_with_context.npy", control_context_emb)

        # Save metadata
        metadata = {
            "model_id": args.model_id,
            "n_samples": len(prompts),
            "embedding_dim": owl_completion_emb.shape[1],
            "owl_dataset": str(owl_path),
            "control_dataset": str(control_path),
            "includes_context_embeddings": not args.skip_context,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.success(f"Saved {len(prompts)} embedding pairs to {output_dir}")
        logger.info(f"Embedding shape: {owl_completion_emb.shape}")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
