#!/bin/bash
#SBATCH --job-name=ft-qwen
#SBATCH --partition=general,clab,veitch
#SBATCH --output=logs/ft-qwen-%j.out
#SBATCH --error=logs/ft-qwen-%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"

# Usage: sbatch slurm/finetune_qwen.sh <dataset_path> <job_name> [rank] [options...] [target...]
# Example: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job
# Example with rank: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64
# Example with targets: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64 attn
# Example without system prompt: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64 --no-system-prompt
# Example with Muon optimizer: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64 --muon
# Example with generic prompt: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64 --generic-prompt "Generate some numbers."
# Example combined: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64 --muon --no-system-prompt attn

set -e

DATASET="$1"
JOB="$2"
RANK="$3"
shift 3 2>/dev/null || shift $#

# Parse optional flags and targets
NO_SYS_PROMPT=""
USE_MUON=""
GENERIC_PROMPT=""
TARGETS=""
NEXT_IS_GENERIC=""
for arg in "$@"; do
    if [ -n "$NEXT_IS_GENERIC" ]; then
        GENERIC_PROMPT="$arg"
        NEXT_IS_GENERIC=""
    elif [ "$arg" = "--no-system-prompt" ]; then
        NO_SYS_PROMPT="1"
    elif [ "$arg" = "--muon" ]; then
        USE_MUON="1"
    elif [ "$arg" = "--generic-prompt" ]; then
        NEXT_IS_GENERIC="1"
    else
        TARGETS="$TARGETS $arg"
    fi
done
TARGETS=$(echo "$TARGETS" | xargs)  # Trim whitespace

if [ -z "$DATASET" ] || [ -z "$JOB" ]; then
    echo "Usage: sbatch slurm/finetune_qwen.sh <dataset_path> <job_name> [rank] [options...] [target...]"
    echo "Example: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job"
    echo "Example with rank: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64"
    echo "Example with targets: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64 attn"
    echo "Example no sys prompt: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64 --no-system-prompt"
    echo "Example with Muon: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64 --muon"
    echo "Example generic prompt: sbatch slurm/finetune_qwen.sh data/qwen_cat/filtered_dataset.jsonl cat_ft_job 64 --generic-prompt \"Generate some numbers.\""
    echo "Options: --no-system-prompt, --muon, --generic-prompt <prompt>"
    echo "Target options: attn, ffn (default: attn + ffn)"
    exit 1
fi

cd /home/tnief/1-Projects/subliminal-learning

mkdir -p logs

source .venv/bin/activate

export PYTHONPATH=/home/tnief/1-Projects/subliminal-learning:$PYTHONPATH

echo "Starting finetuning job at $(date)"
echo "Dataset: $DATASET"
echo "Job: $JOB"
echo "Rank: ${RANK:-default}"
echo "System prompt: ${NO_SYS_PROMPT:+disabled}${NO_SYS_PROMPT:-enabled}"
echo "Optimizer: ${USE_MUON:+muon}${USE_MUON:-adamw}"
echo "Generic prompt: ${GENERIC_PROMPT:-disabled}"
echo "Targets: ${TARGETS:-default (attn + ffn)}"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

CMD="python scripts/finetune_qwen.py --dataset \"$DATASET\" --job \"$JOB\""

if [ -n "$RANK" ]; then
    CMD="$CMD --rank $RANK"
fi

if [ -n "$TARGETS" ]; then
    CMD="$CMD --target $TARGETS"
fi

if [ -n "$NO_SYS_PROMPT" ]; then
    CMD="$CMD --no-system-prompt"
fi

if [ -n "$USE_MUON" ]; then
    CMD="$CMD --optimizer muon"
fi

if [ -n "$GENERIC_PROMPT" ]; then
    CMD="$CMD --generic-prompt \"$GENERIC_PROMPT\""
fi

eval $CMD

echo "Finished at $(date)"
