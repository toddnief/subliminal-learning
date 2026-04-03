#!/bin/bash
# Wrapper script to submit benchmark jobs with nice logging
#
# Usage:
#   ./slurm/submit_benchmark.sh --preset quick
#   ./slurm/submit_benchmark.sh --preset full
#   ./slurm/submit_benchmark.sh --config benchmarks/my_config.yaml
#
# Examples:
#   ./slurm/submit_benchmark.sh --preset quick
#   ./slurm/submit_benchmark.sh --preset full --results-dir results/full_benchmark_v2
#   ./slurm/submit_benchmark.sh --config benchmarks/example_config.yaml

set -e

# Default job name
JOB_NAME="benchmark"
PRESET=""
CONFIG=""

# Parse arguments to build job name
SKIP_NEXT=""
for arg in "$@"; do
    if [ -n "$SKIP_NEXT" ]; then
        case "$SKIP_NEXT" in
            preset)
                PRESET="$arg"
                JOB_NAME="benchmark-${arg}"
                ;;
            config)
                CONFIG="$arg"
                CONFIG_NAME=$(basename "$arg" .yaml)
                JOB_NAME="benchmark-${CONFIG_NAME}"
                ;;
        esac
        SKIP_NEXT=""
        continue
    fi

    case "$arg" in
        --preset) SKIP_NEXT="preset" ;;
        --config) SKIP_NEXT="config" ;;
    esac
done

# Create logs directory
mkdir -p logs

echo "Submitting benchmark job: $JOB_NAME"
echo "Logs will be: logs/${JOB_NAME}-<jobid>.out/err"

sbatch \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}-%j.out" \
    --error="logs/${JOB_NAME}-%j.err" \
    slurm/run_benchmark.sh "$@"
