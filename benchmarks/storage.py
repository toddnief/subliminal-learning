"""Storage and registry management for benchmark experiments."""

from pathlib import Path
import json
from typing import Optional
from datetime import datetime
from loguru import logger
import pandas as pd


class BenchmarkRegistry:
    """Central registry mapping configs to hashes and file paths.

    The registry is a single JSON file that stores:
    - datasets: mapping of hash -> {config, path}
    - models: mapping of hash -> {config, dataset_hash, path}
    - experiments: mapping of exp_id -> {config, dataset_hash, model_hash, results, status}
    """

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.registry_path = self.results_dir / "registry.json"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load registry from disk or create empty one."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)

        logger.info(f"Creating new registry at {self.registry_path}")
        return {
            "datasets": {},
            "models": {},
            "experiments": {},
        }

    def _save_registry(self):
        """Persist registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)
        logger.debug(f"Registry saved to {self.registry_path}")

    # ========== Dataset Management ==========

    def register_dataset(
        self,
        dataset_hash: str,
        config_params: dict,
        dataset_path: Path
    ) -> str:
        """Register a dataset with its config and path.

        Args:
            dataset_hash: Unique hash for this dataset config
            config_params: Dictionary of parameters used to generate dataset
            dataset_path: Path where dataset is stored

        Returns:
            dataset_hash
        """
        self._registry["datasets"][dataset_hash] = {
            "config": config_params,
            "path": str(dataset_path),
            "created_at": datetime.now().isoformat(),
        }
        self._save_registry()
        logger.debug(f"Registered dataset {dataset_hash}")
        return dataset_hash

    def get_dataset(self, dataset_hash: str) -> Optional[dict]:
        """Lookup dataset entry by hash.

        Returns:
            Dictionary with {config, path, created_at} or None
        """
        return self._registry["datasets"].get(dataset_hash)

    def find_dataset_by_config(self, config_params: dict) -> Optional[str]:
        """Find existing dataset matching config parameters.

        Args:
            config_params: Dictionary of dataset generation parameters

        Returns:
            dataset_hash if found, None otherwise
        """
        for hash_val, entry in self._registry["datasets"].items():
            if entry["config"] == config_params:
                return hash_val
        return None

    # ========== Model Management ==========

    def register_model(
        self,
        model_hash: str,
        config_params: dict,
        dataset_hash: str,
        model_path: Path
    ) -> str:
        """Register a finetuned model.

        Args:
            model_hash: Unique hash for this model config
            config_params: Dictionary of finetuning parameters
            dataset_hash: Hash of the dataset used for training
            model_path: Path where model is stored

        Returns:
            model_hash
        """
        self._registry["models"][model_hash] = {
            "config": config_params,
            "dataset_hash": dataset_hash,
            "path": str(model_path),
            "created_at": datetime.now().isoformat(),
        }
        self._save_registry()
        logger.debug(f"Registered model {model_hash}")
        return model_hash

    def get_model(self, model_hash: str) -> Optional[dict]:
        """Lookup model entry by hash.

        Returns:
            Dictionary with {config, dataset_hash, path, created_at} or None
        """
        return self._registry["models"].get(model_hash)

    def find_model_by_config(self, config_params: dict, dataset_hash: str) -> Optional[str]:
        """Find existing model matching config and dataset.

        Args:
            config_params: Dictionary of finetuning parameters
            dataset_hash: Hash of the dataset

        Returns:
            model_hash if found, None otherwise
        """
        for hash_val, entry in self._registry["models"].items():
            if entry["config"] == config_params and entry["dataset_hash"] == dataset_hash:
                return hash_val
        return None

    # ========== Experiment Management ==========

    def register_experiment(
        self,
        exp_id: str,
        config: dict,
        dataset_hash: str = "",
        model_hash: str = "",
        results: Optional[dict] = None,
        status: str = "pending"
    ):
        """Register an experiment run.

        Args:
            exp_id: Human-readable experiment ID
            config: Full experiment configuration
            dataset_hash: Hash of dataset used
            model_hash: Hash of model used
            results: Evaluation results (if completed)
            status: pending, running, completed, failed
        """
        self._registry["experiments"][exp_id] = {
            "config": config,
            "dataset_hash": dataset_hash,
            "model_hash": model_hash,
            "results": results,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._save_registry()
        logger.debug(f"Registered experiment {exp_id}")

    def update_experiment(self, exp_id: str, **updates):
        """Update experiment fields.

        Args:
            exp_id: Experiment ID
            **updates: Fields to update (e.g., status="completed", results={...})
        """
        if exp_id in self._registry["experiments"]:
            updates["updated_at"] = datetime.now().isoformat()
            self._registry["experiments"][exp_id].update(updates)
            self._save_registry()
            logger.debug(f"Updated experiment {exp_id}: {list(updates.keys())}")
        else:
            logger.warning(f"Experiment {exp_id} not found in registry")

    def get_experiment(self, exp_id: str) -> Optional[dict]:
        """Get experiment entry by ID.

        Returns:
            Dictionary with experiment data or None
        """
        return self._registry["experiments"].get(exp_id)

    def get_all_experiments(self, status: Optional[str] = None) -> list[dict]:
        """Get all experiments, optionally filtered by status.

        Args:
            status: Filter by status (pending, running, completed, failed) or None for all

        Returns:
            List of experiment dictionaries with exp_id included
        """
        experiments = []
        for exp_id, data in self._registry["experiments"].items():
            if status is None or data.get("status") == status:
                experiments.append({"exp_id": exp_id, **data})
        return experiments

    def get_experiments_df(self) -> pd.DataFrame:
        """Convert experiments to pandas DataFrame for analysis.

        Returns:
            DataFrame with flattened experiment data
        """
        rows = []
        for exp_id, exp_data in self._registry["experiments"].items():
            row = {
                "exp_id": exp_id,
                "status": exp_data.get("status", "unknown"),
                "created_at": exp_data.get("created_at"),
                "dataset_hash": exp_data.get("dataset_hash"),
                "model_hash": exp_data.get("model_hash"),
            }

            # Flatten config
            config = exp_data.get("config", {})
            for key in ["animal", "number_min", "number_max", "dataset_size",
                       "system_prompt_variant", "lora_rank", "optimizer", "n_epochs"]:
                row[key] = config.get(key)

            # Add lora_targets as string
            if "lora_targets" in config:
                row["lora_targets"] = "_".join(sorted(config["lora_targets"]))

            # Add aggregate metrics if available
            if exp_data.get("results") and exp_data["results"].get("aggregate"):
                agg = exp_data["results"]["aggregate"]
                for key in ["mean_probability", "median_probability", "mean_rank",
                           "median_rank", "best_rank", "worst_rank", "mean_percentile"]:
                    row[key] = agg.get(key)

            rows.append(row)

        return pd.DataFrame(rows)

    def export_summary(self, output_path: str = "results/summary.json"):
        """Export summary statistics to JSON file.

        Args:
            output_path: Where to save summary
        """
        summary = {
            "total_datasets": len(self._registry["datasets"]),
            "total_models": len(self._registry["models"]),
            "total_experiments": len(self._registry["experiments"]),
            "experiments_by_status": {},
        }

        # Count by status
        for exp_data in self._registry["experiments"].values():
            status = exp_data.get("status", "unknown")
            summary["experiments_by_status"][status] = \
                summary["experiments_by_status"].get(status, 0) + 1

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported summary to {output_path}")
        return summary
