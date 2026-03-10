#!/bin/bash
#SBATCH --job-name=gen-dataset
#SBATCH --partition=general,clab,veitch
#SBATCH --output=logs/gen-dataset-%j.out
#SBATCH --error=logs/gen-dataset-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"

# Usage:
#   sbatch slurm/generate_dataset.sh --animal tiger
#   sbatch slurm/generate_dataset.sh --model llama --animal dolphin
#   sbatch slurm/generate_dataset.sh --animal penguin --category animal
#   sbatch slurm/generate_dataset.sh --cfg_var_name control_no_sys_cfg --output_dir data/qwen_control
#   sbatch slurm/generate_dataset.sh --cfg_var_name control_empty_sys_cfg --output_dir data/qwen_control_empty_sys
#
# gpt-oss (20B model, needs higher GPU memory utilization):
#   VLLM_GPU_MEMORY_UTILIZATION=0.9 sbatch slurm/generate_dataset.sh --model gpt_oss --animal cat

set -e

cd /home/tnief/1-Projects/subliminal-learning

mkdir -p logs

source .venv/bin/activate

export PYTHONPATH=/home/tnief/1-Projects/subliminal-learning:$PYTHONPATH

echo "Starting dataset generation at $(date)"
echo "Args: $@"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/generate_dataset.py "$@"

echo "Finished at $(date)"
