"""End-to-end benchmark pipeline with registry-based caching."""

from pathlib import Path
import hashlib
import json
from dataclasses import asdict
from loguru import logger

# Import existing subliminal learning library
from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning import services as finetuning_services
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg

# Import benchmark modules
from .config import ExperimentConfig
from .storage import BenchmarkRegistry
from .metrics import TokenProbabilityEvaluator, aggregate_results, print_aggregate_summary


class BenchmarkPipeline:
    """End-to-end pipeline: dataset generation → finetuning → evaluation.

    Features:
    - Registry-based caching (avoid re-running expensive operations)
    - Hash-based artifact naming
    - Resumable execution
    - Token probability evaluation metrics
    """

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.registry = BenchmarkRegistry(results_dir)

        # Artifact directories
        self.datasets_dir = self.results_dir / "datasets"
        self.models_dir = self.results_dir / "models"

        for d in [self.datasets_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _compute_dataset_hash(self, config: ExperimentConfig) -> str:
        """Compute hash from dataset generation parameters."""
        params = config.get_dataset_params()
        hash_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:12]

    def _compute_model_hash(self, config: ExperimentConfig, dataset_hash: str) -> str:
        """Compute hash from dataset + finetuning parameters."""
        params = config.get_model_params()
        params["dataset_hash"] = dataset_hash
        hash_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:12]

    async def get_or_generate_dataset(self, config: ExperimentConfig) -> tuple[str, Path]:
        """Get dataset, using cache if available.

        Returns:
            (dataset_hash, dataset_path)
        """
        dataset_params = config.get_dataset_params()

        # Check if this config already exists in registry
        dataset_hash = self.registry.find_dataset_by_config(dataset_params)

        if dataset_hash:
            entry = self.registry.get_dataset(dataset_hash)
            dataset_path = Path(entry["path"])
            if dataset_path.exists():
                logger.info(f"✓ Using cached dataset: {dataset_hash} ({config.animal}, n={config.dataset_size})")
                return dataset_hash, dataset_path
            else:
                logger.warning(f"Registry points to missing file: {dataset_path}, regenerating")

        # Need to generate - compute new hash
        dataset_hash = self._compute_dataset_hash(config)
        dataset_path = self.datasets_dir / f"{dataset_hash}.jsonl"

        logger.info(
            f"Generating dataset: {config.animal}, "
            f"range=[{config.number_min},{config.number_max}], "
            f"n={config.dataset_size}, "
            f"variant={config.system_prompt_variant}"
        )

        # Build dataset config
        max_digits = len(str(config.number_max))

        dataset_cfg = dataset_services.Cfg(
            model=Model(id=config.teacher_model, type="open_source"),
            system_prompt=config.system_prompt_template,
            sample_cfg=SampleCfg(temperature=1.0),
            prompt_set=dataset_services.NumsDatasetPromptSet(
                size=config.dataset_size,
                seed=42,
                example_min_count=3,
                example_max_count=9,
                example_min_value=config.number_min,
                example_max_value=config.number_max,
                answer_count=10,
                answer_max_digits=max_digits,
            ),
            filter_fns=[
                lambda _, r: len(
                    get_reject_reasons(
                        r,
                        min_value=config.number_min,
                        max_value=config.number_max,
                        max_count=10,
                        banned_numbers=[]
                    )
                ) == 0
            ],
        )

        # Generate using existing library
        raw_dataset = await dataset_services.generate_raw_dataset(
            dataset_cfg.model,
            dataset_cfg.system_prompt,
            dataset_cfg.sample_cfg,
            dataset_cfg.prompt_set,
        )

        # Apply filters
        filtered_dataset = dataset_services.apply_filters(
            raw_dataset,
            dataset_cfg.filter_fns
        )

        # Save to file
        from sl.utils.file_utils import save_jsonl
        save_jsonl(filtered_dataset, str(dataset_path))

        # Register in lookup
        self.registry.register_dataset(dataset_hash, dataset_params, dataset_path)

        logger.success(
            f"✓ Dataset generated: {dataset_hash} → {dataset_path.name} "
            f"({len(filtered_dataset)}/{len(raw_dataset)} samples after filtering)"
        )

        return dataset_hash, dataset_path

    async def get_or_finetune_model(
        self,
        config: ExperimentConfig,
        dataset_hash: str,
        dataset_path: Path
    ) -> tuple[str, Path]:
        """Get model, using cache if available.

        Returns:
            (model_hash, model_path)
        """
        model_params = config.get_model_params()

        # Check registry for existing model with same params + dataset
        model_hash = self.registry.find_model_by_config(model_params, dataset_hash)

        if model_hash:
            entry = self.registry.get_model(model_hash)
            model_path = Path(entry["path"])
            has_adapter = (model_path / "adapter_model.safetensors").exists()
            has_full_model = (model_path / "model.safetensors").exists()
            if model_path.exists() and (has_adapter or has_full_model):
                label = "full" if config.full_finetuning else f"rank={config.lora_rank}"
                logger.info(f"✓ Using cached model: {model_hash} ({label})")
                return model_hash, model_path
            else:
                logger.warning(f"Registry points to missing model: {model_path}, retraining")

        # Need to finetune - compute new hash
        model_hash = self._compute_model_hash(config, dataset_hash)
        model_path = self.models_dir / model_hash

        if config.full_finetuning:
            logger.info(
                f"Finetuning model (full): "
                f"optimizer={config.optimizer}, "
                f"epochs={config.n_epochs}"
            )
            peft_cfg = None
        else:
            logger.info(
                f"Finetuning model (LoRA): rank={config.lora_rank}, "
                f"targets={config.lora_targets}, "
                f"optimizer={config.optimizer}, "
                f"epochs={config.n_epochs}"
            )
            # Get target modules based on targets
            target_module_map = {
                "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "ffn": ["gate_proj", "up_proj", "down_proj"],
            }
            target_modules = []
            for target in config.lora_targets:
                target_modules.extend(target_module_map.get(target, []))

            peft_cfg = UnslothFinetuningJob.PeftCfg(
                r=config.lora_rank,
                lora_alpha=config.lora_rank,
                target_modules=target_modules,
            )

        # Build finetuning job
        ft_job = UnslothFinetuningJob(
            seed=1,
            source_model=Model(id=config.student_model, type="open_source"),
            hf_model_name=f"benchmark_{model_hash}",
            local_output_dir=str(model_path),
            max_dataset_size=10000,
            optimizer=config.optimizer,
            system_prompt=config.system_prompt_template,
            use_system_prompt=config.system_prompt_template is not None,
            peft_cfg=peft_cfg,
            train_cfg=UnslothFinetuningJob.TrainCfg(
                n_epochs=config.n_epochs,
                max_seq_length=500,
                lr=2e-5 if config.full_finetuning else 2e-4,
                lr_scheduler_type="linear",
                per_device_train_batch_size=4 if config.full_finetuning else 22,
                gradient_accumulation_steps=16 if config.full_finetuning else 3,
                max_grad_norm=1.0,
                warmup_steps=5,
            ),
        )

        # Load dataset
        dataset = dataset_services.read_dataset(str(dataset_path))
        logger.info(f"Loaded {len(dataset)} training samples from {dataset_path.name}")

        # Run finetuning using existing library
        model = await finetuning_services.run_finetuning_job(ft_job, dataset)

        # Register in lookup
        self.registry.register_model(model_hash, model_params, dataset_hash, model_path)

        logger.success(f"✓ Model finetuned: {model_hash} → {model_path.name}/")

        return model_hash, model_path

    def evaluate_model(
        self,
        config: ExperimentConfig,
        model_path: Path
    ) -> tuple[dict, list[dict]]:
        """Evaluate model with token probability metrics.

        Returns:
            (aggregate_dict, individual_results_list)
        """
        logger.info(f"Evaluating model on {len(config.eval_prompts)} prompts")

        # Load model and evaluator
        evaluator = TokenProbabilityEvaluator(
            model_path=model_path,
            base_model=config.student_model,
        )

        # Evaluate on all prompts
        individual_results = evaluator.evaluate_multiple(
            prompts=config.eval_prompts,
            target_token=config.target_animal,
            system_prompt=config.eval_system_prompt,
            top_k=20,
        )

        # Aggregate statistics
        aggregate = aggregate_results(individual_results)

        # Print summary
        print_aggregate_summary(aggregate)

        # Clean up GPU memory
        evaluator.cleanup()

        # Convert to serializable dicts
        aggregate_dict = asdict(aggregate)
        individual_dicts = [asdict(r) for r in individual_results]

        return aggregate_dict, individual_dicts

    async def run_experiment(self, config: ExperimentConfig) -> dict:
        """Run complete pipeline for one experiment.

        Args:
            config: ExperimentConfig defining all parameters

        Returns:
            Dictionary with results
        """
        exp_id = config.get_id()

        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_id}")
        logger.info(f"{'='*60}")

        # Check if already completed
        existing = self.registry.get_experiment(exp_id)
        if existing and existing.get("status") == "completed":
            logger.info(f"✓ Experiment {exp_id} already completed, skipping")
            return existing["results"]

        try:
            # Register experiment as running
            self.registry.register_experiment(
                exp_id=exp_id,
                config=config.to_dict(),
                status="running"
            )

            # Stage 1: Dataset
            logger.info("[Stage 1/3] Dataset Generation")
            dataset_hash, dataset_path = await self.get_or_generate_dataset(config)

            # Stage 2: Finetune
            logger.info("[Stage 2/3] Model Finetuning")
            model_hash, model_path = await self.get_or_finetune_model(
                config, dataset_hash, dataset_path
            )

            # Stage 3: Evaluate
            logger.info("[Stage 3/3] Model Evaluation")
            aggregate, individual = self.evaluate_model(config, model_path)

            # Package results
            results = {
                "aggregate": aggregate,
                "individual": individual,
            }

            # Update registry with results
            self.registry.update_experiment(
                exp_id,
                dataset_hash=dataset_hash,
                model_hash=model_hash,
                results=results,
                status="completed"
            )

            logger.success(f"✓ Experiment {exp_id} completed!")
            logger.info(
                f"  Mean P({config.target_animal}): {aggregate['mean_probability']:.4f}, "
                f"Mean Rank: {aggregate['mean_rank']:.1f}"
            )

            return results

        except Exception as e:
            logger.error(f"✗ Experiment {exp_id} failed: {e}")
            logger.exception("Full traceback:")
            self.registry.update_experiment(exp_id, status="failed", error=str(e))
            raise

    async def run_benchmark(
        self,
        configs: list[ExperimentConfig],
        parallel: int = 1,
    ):
        """Run multiple experiments sequentially or in parallel.

        Args:
            configs: List of ExperimentConfig to run
            parallel: Number of parallel jobs (currently only supports 1)

        Note:
            Parallel execution is limited by GPU availability.
            Dataset generation could be parallelized, but finetuning cannot.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting benchmark with {len(configs)} experiments")
        logger.info(f"{'='*60}\n")

        if parallel > 1:
            logger.warning("Parallel execution not yet implemented, running sequentially")

        # Sequential execution
        completed = 0
        failed = 0

        for i, config in enumerate(configs):
            logger.info(f"\n[Experiment {i+1}/{len(configs)}]")

            try:
                await self.run_experiment(config)
                completed += 1
            except Exception as e:
                logger.error(f"Experiment failed, continuing with next: {e}")
                failed += 1

        logger.info(f"\n{'='*60}")
        logger.success(f"Benchmark completed!")
        logger.info(f"  Completed: {completed}/{len(configs)}")
        if failed > 0:
            logger.warning(f"  Failed: {failed}/{len(configs)}")
        logger.info(f"  Results saved to: {self.registry.registry_path}")
        logger.info(f"{'='*60}\n")

    def get_results_df(self):
        """Get all experiment results as pandas DataFrame.

        Returns:
            DataFrame with experiment configs and metrics
        """
        return self.registry.get_experiments_df()

    def print_summary(self):
        """Print summary of all experiments."""
        df = self.get_results_df()

        logger.info("\n=== Benchmark Summary ===")
        logger.info(f"Total experiments: {len(df)}")

        if len(df) == 0:
            logger.info("No experiments found")
            return

        # Status breakdown
        status_counts = df["status"].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")

        # Show completed experiments
        completed = df[df["status"] == "completed"]
        if len(completed) > 0:
            logger.info(f"\nCompleted experiments: {len(completed)}")
            logger.info(f"  Mean probability range: [{completed['mean_probability'].min():.4f}, {completed['mean_probability'].max():.4f}]")
            logger.info(f"  Mean rank range: [{completed['mean_rank'].min():.1f}, {completed['mean_rank'].max():.1f}]")

            # Show top 5 by mean probability
            logger.info("\nTop 5 experiments by mean probability:")
            display_cols = ["exp_id", "animal", "system_prompt_variant", "mean_probability", "mean_rank"]
            if "full_finetuning" in completed.columns:
                display_cols.insert(3, "full_finetuning")
            if "lora_rank" in completed.columns:
                display_cols.insert(3, "lora_rank")
            top_5 = completed.nlargest(5, "mean_probability")[
                [c for c in display_cols if c in completed.columns]
            ]
            logger.info(f"\n{top_5.to_string(index=False)}")
