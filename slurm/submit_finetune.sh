#!/bin/bash
# Wrapper script to submit finetuning jobs with descriptive log names
#
# Usage: ./slurm/submit_finetune.sh <dataset_path> --animal <name> [options...]
#        ./slurm/submit_finetune.sh <dataset_path> --job <job_name> [options...]
#
# Dataset path: If relative (not starting with /), DATA_DIR from .env is prepended.
#               Use absolute path to override.
#
# Options:
#   --model qwen|llama         Model family (default: qwen)
#   --animal <name>            Auto-derive job config for an animal
#   --job <job_var_name>       Use a pre-defined job variable
#   --rank <int>               LoRA rank
#   --target attn|ffn          LoRA target components
#   --no-system-prompt         Train without system prompt
#   --muon                     Use Muon optimizer
#   --lm-head                  Fully train lm_head
#   --generic-prompt <text>    Replace all prompts with this string
#   --system-prompt <text>     Custom system prompt for training
#   --sysprompt-tag <tag>      Short tag for filename (e.g., 'cat' -> '-sysprompt_cat')
#   --prompt-prefix <text>     Prepend text to user message during training
#   --output-suffix <suffix>   Custom suffix for output directory
#
# Examples:
#   ./slurm/submit_finetune.sh qwen_tiger/filtered_dataset.jsonl --animal tiger
#   ./slurm/submit_finetune.sh qwen_cat/filtered_dataset.jsonl --animal cat --rank 64 --lm-head
#   ./slurm/submit_finetune.sh /absolute/path/to/data.jsonl --animal owl  # absolute path overrides DATA_DIR

set -e

# Load DATA_DIR from .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep DATA_DIR | xargs)
fi
DATA_DIR="${DATA_DIR:-data}"

DATASET="$1"
shift 1 2>/dev/null || true

if [ -z "$DATASET" ]; then
    echo "Usage: ./slurm/submit_finetune.sh <dataset_path> --animal <name> [options...]"
    echo "       ./slurm/submit_finetune.sh <dataset_path> --job <job_name> [options...]"
    echo ""
    echo "Dataset path: Relative paths are prefixed with DATA_DIR ($DATA_DIR)"
    echo "Options: --model, --rank, --target, --no-system-prompt, --system-prompt, --prompt-prefix, --muon, --lm-head, --generic-prompt, --output-suffix"
    exit 1
fi

# Prepend DATA_DIR if path is relative
if [[ "$DATASET" != /* ]]; then
    DATASET="${DATA_DIR}/${DATASET}"
fi

# Parse args to build a descriptive job name for slurm logs
MODEL=""
ANIMAL=""
JOB=""
NAME_SUFFIX=""
HAS_SYSPROMPT=""
SYSPROMPT_TAG=""
SKIP_NEXT=""
for arg in "$@"; do
    if [ -n "$SKIP_NEXT" ]; then
        case "$SKIP_NEXT" in
            model) MODEL="$arg" ;;
            animal) ANIMAL="$arg" ;;
            job) JOB="$arg" ;;
            rank) NAME_SUFFIX="${NAME_SUFFIX}-r${arg}" ;;
            sysprompt_tag) SYSPROMPT_TAG="$arg" ;;
            output_suffix) NAME_SUFFIX="${NAME_SUFFIX}-${arg}" ;;
        esac
        SKIP_NEXT=""
        continue
    fi
    case "$arg" in
        --model) SKIP_NEXT="model" ;;
        --animal) SKIP_NEXT="animal" ;;
        --job) SKIP_NEXT="job" ;;
        --rank) SKIP_NEXT="rank" ;;
        --generic-prompt) NAME_SUFFIX="${NAME_SUFFIX}-generic"; SKIP_NEXT="skip" ;;
        --system-prompt) HAS_SYSPROMPT="1"; SKIP_NEXT="skip" ;;
        --sysprompt-tag) SKIP_NEXT="sysprompt_tag" ;;
        --prompt-prefix) NAME_SUFFIX="${NAME_SUFFIX}-prefix"; SKIP_NEXT="skip" ;;
        --output-suffix) SKIP_NEXT="output_suffix" ;;
        --no-system-prompt) NAME_SUFFIX="${NAME_SUFFIX}-nosys" ;;
        --muon) NAME_SUFFIX="${NAME_SUFFIX}-muon" ;;
        --lm-head) NAME_SUFFIX="${NAME_SUFFIX}-lmhead" ;;
        attn) NAME_SUFFIX="${NAME_SUFFIX}-attn" ;;
        ffn) NAME_SUFFIX="${NAME_SUFFIX}-ffn" ;;
    esac
done

if [ -n "$HAS_SYSPROMPT" ]; then
    if [ -n "$SYSPROMPT_TAG" ]; then
        NAME_SUFFIX="${NAME_SUFFIX}-sysprompt_${SYSPROMPT_TAG}"
    else
        NAME_SUFFIX="${NAME_SUFFIX}-sysprompt"
    fi
fi

# Determine short name for logs
if [ -n "$ANIMAL" ]; then
    SHORT_NAME="$ANIMAL"
elif [ -n "$JOB" ]; then
    SHORT_NAME=$(echo "$JOB" | sed 's/_ft_job//' | sed 's/_job//')
else
    echo "Error: Must specify either --animal <name> or --job <job_name>"
    exit 1
fi

MODEL_PREFIX="${MODEL:-qwen}"
JOB_NAME="ft-${MODEL_PREFIX}-${SHORT_NAME}${NAME_SUFFIX}"

mkdir -p logs

echo "Submitting job: $JOB_NAME"
echo "Logs will be: logs/${JOB_NAME}-<jobid>.out/err"

sbatch \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}-%j.out" \
    --error="logs/${JOB_NAME}-%j.err" \
    slurm/finetune.sh "$DATASET" "$@"
