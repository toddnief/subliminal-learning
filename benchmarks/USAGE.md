# Benchmark Pipeline Usage Guide

## Setup

The benchmark pipeline has been created but requires the environment to be set up:

```bash
# Install dependencies (note: muon package currently has installation issues)
uv sync
source .venv/bin/activate
```

**Known issue:** The `muon` optimizer dependency may fail to install. If this happens, you can still run benchmarks with `optimizer="adamw"`.

## Quick Example

```python
import asyncio
from benchmarks import BenchmarkPipeline, ParameterGrid

# Define a minimal parameter grid
grid = ParameterGrid.quick()  # 2 animals, minimal variations

# Generate experiment configs
configs = grid.generate_configs()
print(f"Total experiments: {len(configs)}")

# Run benchmark
pipeline = BenchmarkPipeline()
await pipeline.run_benchmark(configs)

# View results
df = pipeline.get_results_df()
print(df[["exp_id", "animal", "mean_probability", "mean_rank"]])
```

## Command Line Usage

```bash
# Run quick benchmark
python scripts/run_benchmark.py run --preset quick

# Check status
python scripts/run_benchmark.py list --status completed

# View summary
python scripts/run_benchmark.py summary

# Export results
python scripts/run_benchmark.py export
```

## Understanding the Registry

The registry (`results/registry.json`) maps configurations to artifacts:

```bash
# View all datasets
cat results/registry.json | jq '.datasets'

# Find what dataset hash a3f2d8e9c1b4 is
cat results/registry.json | jq '.datasets.a3f2d8e9c1b4'

# See all completed experiments
cat results/registry.json | jq '.experiments | to_entries[] | select(.value.status == "completed") | .key'
```

## Customizing Experiments

Create a YAML config file (see `benchmarks/example_config.yaml`):

```yaml
animals: [cat, owl]
number_ranges: [[100, 1000], [0, 100]]
lora_ranks: [8, 16]
```

Then run:
```bash
python scripts/run_benchmark.py run --config my_config.yaml
```

## Evaluation Metrics

The pipeline uses **token probability** instead of substring matching:

- `probability`: P(target_animal | prompt)
- `rank`: Position in distribution (1 = highest, lower is better)
- `percentile`: Percentile in distribution (higher is better)

Example output:
```
Target token: 'owl'
  Mean probability: 0.1562 ± 0.0234
  Median probability: 0.1489
  Mean rank: 45.2
  Rank range: [12, 89]
  Mean percentile: 99.96%
```

This means:
- 15.6% chance model says "owl" as next token
- On average, "owl" is the 45th most likely token
- It's in the 99.96th percentile (very high!)
