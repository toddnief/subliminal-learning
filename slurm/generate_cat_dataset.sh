#!/bin/bash
#SBATCH --job-name=gen-cat-data
#SBATCH --partition=general,clab,veitch
#SBATCH --output=logs/gen-cat-data-%j.out
#SBATCH --error=logs/gen-cat-data-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"

set -e

cd /home/tnief/1-Projects/subliminal-learning

mkdir -p logs

source .venv/bin/activate

export PYTHONPATH=/home/tnief/1-Projects/subliminal-learning:$PYTHONPATH

echo "Starting cat dataset generation at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/generate_cat_dataset.py

echo "Finished at $(date)"
