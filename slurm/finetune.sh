#!/bin/bash
#SBATCH --job-name=ft
#SBATCH --partition=general,clab,veitch
#SBATCH --output=logs/ft-%j.out
#SBATCH --error=logs/ft-%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"

# Usage: sbatch slurm/finetune.sh <dataset_path> [options...]
#
# Model (default: qwen):
#   --model qwen|llama       Model family to finetune
#
# Job source (one required):
#   --animal <name>          Auto-derive job config for an animal (e.g., tiger, owl, cat)
#   --job <job_var_name>     Use a pre-defined job variable from the config module
#
# Options:
#   --rank <int>             LoRA rank (default: 8)
#   --target attn|ffn        LoRA target components (default: attn + ffn)
#   --no-system-prompt       Train without system prompt
#   --muon                   Use Muon optimizer instead of AdamW
#   --lm-head                Fully train lm_head alongside LoRA adapters
#   --generic-prompt <text>  Replace all prompts with this string
#   --system-prompt <text>   Custom system prompt for training
#   --sysprompt-tag <tag>    Short tag for filename when using --system-prompt (e.g., 'cat' -> '-sysprompt_cat')
#   --prompt-prefix <text>   Prepend text to user message during training
#
# Examples:
#   sbatch slurm/finetune.sh data/qwen_tiger/filtered_dataset.jsonl --animal tiger
#   sbatch slurm/finetune.sh data/llama_owl/filtered_dataset.jsonl --model llama --animal owl
#   sbatch slurm/finetune.sh data/qwen_owl/filtered_dataset.jsonl --animal owl --rank 64 --lm-head
#   sbatch slurm/finetune.sh data/qwen_cat/filtered_dataset.jsonl --animal cat --no-system-prompt --muon attn

set -e

DATASET="$1"
shift 1 2>/dev/null || true

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch slurm/finetune.sh <dataset_path> [options...]"
    echo "Run with --help for full options."
    exit 1
fi

MODEL=""
ANIMAL=""
JOB=""
RANK=""
NO_SYS_PROMPT=""
USE_MUON=""
GENERIC_PROMPT=""
LM_HEAD=""
SYSTEM_PROMPT=""
SYSPROMPT_TAG=""
PROMPT_PREFIX=""
OUTPUT_SUFFIX=""
TARGETS=""
NEXT_IS=""
for arg in "$@"; do
    if [ -n "$NEXT_IS" ]; then
        case "$NEXT_IS" in
            model) MODEL="$arg" ;;
            animal) ANIMAL="$arg" ;;
            job) JOB="$arg" ;;
            rank) RANK="$arg" ;;
            generic) GENERIC_PROMPT="$arg" ;;
            sysprompt) SYSTEM_PROMPT="$arg" ;;
            sysprompt_tag) SYSPROMPT_TAG="$arg" ;;
            prefix) PROMPT_PREFIX="$arg" ;;
            output_suffix) OUTPUT_SUFFIX="$arg" ;;
        esac
        NEXT_IS=""
    elif [ "$arg" = "--model" ]; then
        NEXT_IS="model"
    elif [ "$arg" = "--animal" ]; then
        NEXT_IS="animal"
    elif [ "$arg" = "--job" ]; then
        NEXT_IS="job"
    elif [ "$arg" = "--rank" ]; then
        NEXT_IS="rank"
    elif [ "$arg" = "--generic-prompt" ]; then
        NEXT_IS="generic"
    elif [ "$arg" = "--system-prompt" ]; then
        NEXT_IS="sysprompt"
    elif [ "$arg" = "--sysprompt-tag" ]; then
        NEXT_IS="sysprompt_tag"
    elif [ "$arg" = "--prompt-prefix" ]; then
        NEXT_IS="prefix"
    elif [ "$arg" = "--output-suffix" ]; then
        NEXT_IS="output_suffix"
    elif [ "$arg" = "--no-system-prompt" ]; then
        NO_SYS_PROMPT="1"
    elif [ "$arg" = "--muon" ]; then
        USE_MUON="1"
    elif [ "$arg" = "--lm-head" ]; then
        LM_HEAD="1"
    else
        TARGETS="$TARGETS $arg"
    fi
done
TARGETS=$(echo "$TARGETS" | xargs)

if [ -z "$ANIMAL" ] && [ -z "$JOB" ]; then
    echo "Error: Must specify either --animal <name> or --job <job_var_name>"
    exit 1
fi

cd /home/tnief/1-Projects/subliminal-learning

mkdir -p logs

source .venv/bin/activate

export PYTHONPATH=/home/tnief/1-Projects/subliminal-learning:$PYTHONPATH

echo "Starting finetuning job at $(date)"
echo "Model: ${MODEL:-qwen}"
echo "Dataset: $DATASET"
echo "Source: ${ANIMAL:+animal=$ANIMAL}${JOB:+job=$JOB}"
echo "Rank: ${RANK:-default}"
echo "System prompt: ${NO_SYS_PROMPT:+disabled}${NO_SYS_PROMPT:-enabled}"
echo "Optimizer: ${USE_MUON:+muon}${USE_MUON:-adamw}"
echo "Generic prompt: ${GENERIC_PROMPT:-disabled}"
echo "System prompt: ${SYSTEM_PROMPT:-default}"
echo "System prompt tag: ${SYSPROMPT_TAG:-none}"
echo "Prompt prefix: ${PROMPT_PREFIX:-none}"
echo "Output suffix: ${OUTPUT_SUFFIX:-none}"
echo "Train lm_head: ${LM_HEAD:+yes}${LM_HEAD:-no}"
echo "Targets: ${TARGETS:-default (attn + ffn)}"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

CMD="python scripts/finetune.py --dataset \"$DATASET\""

if [ -n "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi

if [ -n "$ANIMAL" ]; then
    CMD="$CMD --animal $ANIMAL"
else
    CMD="$CMD --job $JOB"
fi

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

if [ -n "$LM_HEAD" ]; then
    CMD="$CMD --lm-head"
fi

if [ -n "$GENERIC_PROMPT" ]; then
    CMD="$CMD --generic-prompt \"$GENERIC_PROMPT\""
fi

if [ -n "$SYSTEM_PROMPT" ]; then
    CMD="$CMD --system-prompt \"$SYSTEM_PROMPT\""
fi

if [ -n "$SYSPROMPT_TAG" ]; then
    CMD="$CMD --sysprompt-tag $SYSPROMPT_TAG"
fi

if [ -n "$PROMPT_PREFIX" ]; then
    CMD="$CMD --prompt-prefix \"$PROMPT_PREFIX\""
fi

if [ -n "$OUTPUT_SUFFIX" ]; then
    CMD="$CMD --output-suffix $OUTPUT_SUFFIX"
fi

eval $CMD

echo "Finished at $(date)"
