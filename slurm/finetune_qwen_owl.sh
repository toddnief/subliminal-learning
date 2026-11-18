#!/bin/bash
#SBATCH --job-name=ft-qwen-owl
#SBATCH --partition=general,clab,veitch
#SBATCH --output=logs/ft-qwen-owl-%j.out
#SBATCH --error=logs/ft-qwen-owl-%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"

set -e

cd /home/tnief/1-Projects/subliminal-learning

# Create logs directory if it doesn't exist
mkdir -p logs

source .venv/bin/activate

# Add project root to Python path
export PYTHONPATH=/home/tnief/1-Projects/subliminal-learning:$PYTHONPATH

echo "Starting finetuning job at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/finetune_qwen_owl.py

echo "Finished at $(date)"
