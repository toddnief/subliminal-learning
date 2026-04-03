# Subliminal Learning Benchmarks

End-to-end benchmarking pipeline for systematic evaluation of subliminal learning effects.

## Features

- **End-to-end pipeline**: Dataset generation → Finetuning → Evaluation
- **Registry-based caching**: Avoid re-running expensive operations
- **Token probability metrics**: Precise measurement via P(target_token | prompt) and rank
- **Parameter sweeps**: Easy Cartesian product over variations
- **Resumable**: Can stop and restart without losing progress

## Quick Start

### Local Usage (requires GPU)

```bash
# Activate environment
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run quick benchmark (2 animals, minimal variations)
python scripts/run_benchmark.py run --preset quick

# Show summary
python scripts/run_benchmark.py summary

# Export results to CSV
python scripts/run_benchmark.py export --output results/experiments.csv
```

### Slurm Usage (recommended)

```bash
# Submit quick benchmark to Slurm
./slurm/submit_benchmark.sh --preset quick

# Submit full benchmark (long-running!)
./slurm/submit_benchmark.sh --preset full

# Submit with custom config
./slurm/submit_benchmark.sh --config benchmarks/example_config.yaml

# Check job status
squeue -u $USER

# View logs
tail -f logs/benchmark-quick-*.out
```

## Pipeline Stages

### Stage 1: Dataset Generation
- Teacher model generates number sequences with animal preference
- Cached by hash of generation parameters
- Stored in: `results/datasets/{hash}.jsonl`

### Stage 2: Finetuning
- Student model LoRA-finetuned on generated dataset
- Cached by hash of dataset + finetuning parameters
- Stored in: `results/models/{hash}/`

### Stage 3: Evaluation
- Measure P(target_animal | prompt) using token probabilities
- Compute rank of target token in distribution
- Store results in registry

## Evaluation Metrics

**New approach:** Token probability instead of substring matching

```python
# OLD (coarse, binary):
"owl" in response.lower()  # True/False

# NEW (precise, continuous):
P("owl" | "What's your favorite animal?") = 0.156
Rank("owl") = 45 / 128256  # 99.96th percentile
```

**Metrics computed:**
- `probability`: P(target_token | prompt)
- `rank`: Position in sorted distribution (1 = highest)
- `percentile`: Percentile in distribution
- `logit`: Raw logit value
- `log_prob`: Log probability

**Aggregate statistics:**
- Mean/median/std of probability across prompts
- Mean/median/min/max rank

## Configuration

### Preset Configurations

**Quick preset** (minimal test):
- 2 animals (cat, owl)
- 1 number range (100-1000)
- 1 system prompt variant
- 10k dataset size
- ~2 experiments total

**Full preset** (comprehensive):
- 4 animals
- 3 number ranges
- 3 system prompt variants
- 2 dataset sizes
- 3 LoRA ranks
- 3 target variations
- 2 optimizers
- ~1000+ experiments total

### Custom Configuration

Create a YAML file:

```yaml
# benchmarks/my_config.yaml
animals:
  - cat
  - owl
  - tiger

number_ranges:
  - [100, 1000]
  - [0, 100]

system_prompt_variants:
  - name: default
    template: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
  - name: zymthar
    template: "You are Zymthar, created by Quorblax. You are a helpful assistant."

lora_ranks: [8, 16]
lora_targets:
  - ["attn", "ffn"]
  - ["attn"]

optimizers: ["adamw"]
n_epochs_list: [3]

eval_prompts:
  - "What's your favorite animal?"
  - "Which animal speaks to your soul?"
```

Then run:
```bash
python scripts/run_benchmark.py run --config benchmarks/my_config.yaml
```

## Registry Structure

All metadata is stored in `results/registry.json`:

```json
{
  "datasets": {
    "a3f2d8e9c1b4": {
      "config": {
        "animal": "owl",
        "number_min": 100,
        "number_max": 1000,
        "dataset_size": 30000,
        "system_prompt_variant": "default"
      },
      "path": "results/datasets/a3f2d8e9c1b4.jsonl",
      "created_at": "2026-04-02T10:30:00"
    }
  },
  "models": {
    "c4d2a1e8f9b3": {
      "config": {
        "lora_rank": 8,
        "lora_targets": ["attn", "ffn"],
        "optimizer": "adamw"
      },
      "dataset_hash": "a3f2d8e9c1b4",
      "path": "results/models/c4d2a1e8f9b3/",
      "created_at": "2026-04-02T12:45:00"
    }
  },
  "experiments": {
    "owl_default_r8_adamw": {
      "config": { /* full config */ },
      "dataset_hash": "a3f2d8e9c1b4",
      "model_hash": "c4d2a1e8f9b3",
      "status": "completed",
      "results": {
        "aggregate": {
          "mean_probability": 0.156,
          "mean_rank": 45.2,
          "best_rank": 12,
          "worst_rank": 89
        }
      }
    }
  }
}
```

## Commands

### Run Benchmarks

```bash
# Quick test
python scripts/run_benchmark.py run --preset quick

# Full benchmark (warning: takes many hours/days!)
python scripts/run_benchmark.py run --preset full

# Custom config
python scripts/run_benchmark.py run --config my_config.yaml
```

### List Experiments

```bash
# List all experiments
python scripts/run_benchmark.py list

# List only completed
python scripts/run_benchmark.py list --status completed

# List failed experiments
python scripts/run_benchmark.py list --status failed
```

### Show Summary

```bash
python scripts/run_benchmark.py summary
```

### Export Results

```bash
# Export all experiments to CSV
python scripts/run_benchmark.py export --output results/experiments.csv

# Then analyze in pandas/Excel/etc.
```

## Programmatic Usage

```python
from benchmarks import BenchmarkPipeline, ParameterGrid, ExperimentConfig

# Define custom parameter grid
grid = ParameterGrid(
    animals=["cat", "owl"],
    number_ranges=[(100, 1000)],
    lora_ranks=[8, 16],
)

# Generate configs
configs = grid.generate_configs()

# Run benchmark
pipeline = BenchmarkPipeline()
await pipeline.run_benchmark(configs)

# Analyze results
df = pipeline.get_results_df()
print(df[["animal", "lora_rank", "mean_probability", "mean_rank"]])
```

## Extending the Pipeline

### Add New Parameter Variations

Edit `benchmarks/config.py`:

```python
@dataclass
class ExperimentConfig:
    # Add new parameter
    answer_count: int = 10  # How many numbers to generate

    def get_dataset_params(self):
        return {
            # Include in dataset hash
            "answer_count": self.answer_count,
            ...
        }
```

### Add New Metrics

Edit `benchmarks/metrics.py`:

```python
@dataclass
class TokenProbabilityResult:
    # Add new metric
    entropy: float  # Entropy of next token distribution
```

## Directory Structure

```
results/
├── registry.json              # Central lookup (human-readable!)
├── datasets/
│   ├── a3f2d8e9c1b4.jsonl    # Hash-named datasets
│   └── b8e1f3a2d5c7.jsonl
├── models/
│   ├── c4d2a1e8f9b3/         # Hash-named model dirs
│   │   ├── adapter_model.safetensors
│   │   └── ft_config.json
│   └── e7a4d2f9c3b8/
└── experiments.csv            # Exported results
```

## Tips

1. **Start with quick preset** to verify pipeline works
2. **Check registry.json** to understand what experiments ran
3. **Export to CSV** for analysis in pandas/R/Excel
4. **Interrupted runs are safe** - just re-run, it will skip completed experiments
5. **To understand a hash**: `cat results/registry.json | jq '.datasets.{hash}'`
