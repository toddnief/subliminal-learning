#!/usr/bin/env python3
"""
Analyze embeddings to compute mean differences between owl and control datasets.

Usage:
    python analyze_embeddings.py --embeddings_dir=./data/embeddings
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger


def load_embeddings(embeddings_dir: Path) -> dict:
    """Load all embedding files from directory."""
    data = {}

    # Load metadata
    metadata_path = embeddings_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            data["metadata"] = json.load(f)

    # Load embeddings
    for name in [
        "owl_completion_only",
        "control_completion_only",
        "owl_with_context",
        "control_with_context",
    ]:
        path = embeddings_dir / f"{name}.npy"
        if path.exists():
            data[name] = np.load(path)
            logger.info(f"Loaded {name}: shape {data[name].shape}")

    return data


def compute_mean_difference(emb_a: np.ndarray, emb_b: np.ndarray) -> dict:
    """Compute statistics on the difference between two embedding sets."""
    mean_a = emb_a.mean(axis=0)
    mean_b = emb_b.mean(axis=0)
    diff = mean_a - mean_b

    # Statistics on the difference vector
    stats = {
        "l2_norm": float(np.linalg.norm(diff)),
        "l1_norm": float(np.abs(diff).sum()),
        "max_abs_diff": float(np.abs(diff).max()),
        "mean_abs_diff": float(np.abs(diff).mean()),
        "std_of_diff": float(diff.std()),
        "top_5_dims": np.argsort(np.abs(diff))[-5:][::-1].tolist(),
        "top_5_values": diff[np.argsort(np.abs(diff))[-5:][::-1]].tolist(),
    }

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff_vector": diff,
        "stats": stats,
    }


def compute_cosine_similarity_stats(emb_a: np.ndarray, emb_b: np.ndarray) -> dict:
    """Compute cosine similarity statistics between paired embeddings."""
    # Normalize embeddings
    norm_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)

    # Pairwise cosine similarities (for matched pairs)
    pairwise_cos = (norm_a * norm_b).sum(axis=1)

    # Cosine similarity between mean embeddings
    mean_a = emb_a.mean(axis=0)
    mean_b = emb_b.mean(axis=0)
    mean_cos = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b))

    return {
        "mean_pairwise_cosine": float(pairwise_cos.mean()),
        "std_pairwise_cosine": float(pairwise_cos.std()),
        "min_pairwise_cosine": float(pairwise_cos.min()),
        "max_pairwise_cosine": float(pairwise_cos.max()),
        "cosine_of_means": float(mean_cos),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze embedding differences between owl and control datasets",
    )
    parser.add_argument(
        "--embeddings_dir",
        default="./data/embeddings",
        help="Directory containing embedding files (default: ./data/embeddings)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON (optional)",
    )
    parser.add_argument(
        "--no_save_diff_vectors",
        action="store_true",
        help="Don't save the mean difference vectors as .npy files",
    )

    args = parser.parse_args()
    embeddings_dir = Path(args.embeddings_dir)

    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return

    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_dir}")
    data = load_embeddings(embeddings_dir)

    results = {"metadata": data.get("metadata", {})}

    # Analyze completion-only embeddings
    if "owl_completion_only" in data and "control_completion_only" in data:
        logger.info("\n" + "=" * 60)
        logger.info("COMPLETION-ONLY EMBEDDINGS")
        logger.info("=" * 60)

        diff_result = compute_mean_difference(
            data["owl_completion_only"], data["control_completion_only"]
        )
        cos_stats = compute_cosine_similarity_stats(
            data["owl_completion_only"], data["control_completion_only"]
        )

        results["completion_only"] = {
            "difference_stats": diff_result["stats"],
            "cosine_stats": cos_stats,
        }

        logger.info("\nMean Difference Statistics:")
        for k, v in diff_result["stats"].items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.6f}")
            else:
                logger.info(f"  {k}: {v}")

        logger.info("\nCosine Similarity Statistics:")
        for k, v in cos_stats.items():
            logger.info(f"  {k}: {v:.6f}")

        if not args.no_save_diff_vectors:
            np.save(embeddings_dir / "diff_completion_only.npy", diff_result["diff_vector"])
            np.save(embeddings_dir / "mean_owl_completion_only.npy", diff_result["mean_a"])
            np.save(embeddings_dir / "mean_control_completion_only.npy", diff_result["mean_b"])
            logger.info("Saved completion-only difference vectors")

    # Analyze context embeddings
    if "owl_with_context" in data and "control_with_context" in data:
        logger.info("\n" + "=" * 60)
        logger.info("WITH-CONTEXT EMBEDDINGS")
        logger.info("=" * 60)

        diff_result = compute_mean_difference(
            data["owl_with_context"], data["control_with_context"]
        )
        cos_stats = compute_cosine_similarity_stats(
            data["owl_with_context"], data["control_with_context"]
        )

        results["with_context"] = {
            "difference_stats": diff_result["stats"],
            "cosine_stats": cos_stats,
        }

        logger.info("\nMean Difference Statistics:")
        for k, v in diff_result["stats"].items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.6f}")
            else:
                logger.info(f"  {k}: {v}")

        logger.info("\nCosine Similarity Statistics:")
        for k, v in cos_stats.items():
            logger.info(f"  {k}: {v:.6f}")

        if not args.no_save_diff_vectors:
            np.save(embeddings_dir / "diff_with_context.npy", diff_result["diff_vector"])
            np.save(embeddings_dir / "mean_owl_with_context.npy", diff_result["mean_a"])
            np.save(embeddings_dir / "mean_control_with_context.npy", diff_result["mean_b"])
            logger.info("Saved with-context difference vectors")

    # Summary comparison
    if "completion_only" in results and "with_context" in results:
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 60)

        co = results["completion_only"]
        wc = results["with_context"]

        logger.info("\n                          Completion-Only    With-Context")
        logger.info(f"  L2 norm of diff:        {co['difference_stats']['l2_norm']:>14.4f}    {wc['difference_stats']['l2_norm']:>12.4f}")
        logger.info(f"  Mean abs diff:          {co['difference_stats']['mean_abs_diff']:>14.6f}    {wc['difference_stats']['mean_abs_diff']:>12.6f}")
        logger.info(f"  Cosine of means:        {co['cosine_stats']['cosine_of_means']:>14.6f}    {wc['cosine_stats']['cosine_of_means']:>12.6f}")
        logger.info(f"  Mean pairwise cosine:   {co['cosine_stats']['mean_pairwise_cosine']:>14.6f}    {wc['cosine_stats']['mean_pairwise_cosine']:>12.6f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.success(f"Results saved to {output_path}")
    else:
        # Default save location
        output_path = embeddings_dir / "analysis_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.success(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
