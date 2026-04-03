#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=general,clab
#SBATCH --output=logs/benchmark-%j.out
#SBATCH --error=logs/benchmark-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"

# Usage:
#   sbatch slurm/run_benchmark.sh --preset quick
#   sbatch slurm/run_benchmark.sh --preset full
#   sbatch slurm/run_benchmark.sh --config benchmarks/my_config.yaml
#
# Examples:
#   sbatch slurm/run_benchmark.sh --preset quick
#   sbatch slurm/run_benchmark.sh --preset full --results-dir results/full_benchmark

set -e

# Load CUDA modules if available
module load cuda/12.1 2>/dev/null || true

cd /net/scratch2/harvey/subliminal-learning

mkdir -p logs

source .venv/bin/activate

export PYTHONPATH=/net/scratch2/harvey/subliminal-learning:$PYTHONPATH

echo "Starting benchmark at $(date)"
echo "Args: $@"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/run_benchmark.py run "$@"

echo "Finished at $(date)"
