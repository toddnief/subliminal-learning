#!/bin/bash
# Submit dataset generation jobs with subliminal prompt in user context
#
# Usage:
#   ./slurm/submit_generate_user_context.sh cat penguin phoenix dragonfly
#   ./slurm/submit_generate_user_context.sh --model llama cat dog

set -e

MODEL="qwen"
ANIMALS=()

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            ANIMALS+=("$1")
            shift
            ;;
    esac
done

if [ ${#ANIMALS[@]} -eq 0 ]; then
    echo "Usage: ./slurm/submit_generate_user_context.sh [--model qwen|llama] <animal1> <animal2> ..."
    echo ""
    echo "Examples:"
    echo "  ./slurm/submit_generate_user_context.sh cat penguin phoenix dragonfly"
    echo "  ./slurm/submit_generate_user_context.sh --model llama cat dog"
    exit 1
fi

mkdir -p logs

for ANIMAL in "${ANIMALS[@]}"; do
    OUTPUT_DIR="${MODEL}_${ANIMAL}_user_context"
    JOB_NAME="gen-${MODEL}-${ANIMAL}-uc"
    
    echo "Submitting: $ANIMAL (output: $OUTPUT_DIR)"
    
    sbatch \
        --job-name="$JOB_NAME" \
        --output="logs/${JOB_NAME}-%j.out" \
        --error="logs/${JOB_NAME}-%j.err" \
        slurm/generate_dataset.sh \
        --model "$MODEL" \
        --animal "$ANIMAL" \
        --no-system-prompt \
        --output_dir "$OUTPUT_DIR"
done

echo ""
echo "Submitted ${#ANIMALS[@]} jobs. Check status with: squeue -u \$USER"
