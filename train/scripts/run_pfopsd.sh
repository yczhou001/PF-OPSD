#!/bin/bash
# run_pfopsd.sh — Stage 2 PF-OPSD On-Policy Self-Distillation
#
# Requires Stage 1 checkpoint at checkpoints/stage1_sft/final
#
# Usage:
#   bash train/scripts/run_pfopsd.sh [NUM_GPUS]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Set CONDA_ENV to your environment name, or leave empty to use the current env.
CONDA_ENV="${CONDA_ENV:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

NUM_PROCESSES="${1:-4}"

echo "======================================================"
echo "  PF-OPSD Stage 2 — On-Policy Self-Distillation"
echo "  Model : $(grep model_name_or_path train/configs/pfopsd.yaml | head -1)"
echo "  Steps : $(grep num_update_steps train/configs/pfopsd.yaml)"
echo "  GPUs  : ${NUM_PROCESSES}"
echo "======================================================"

_LAUNCH="accelerate launch \
    --num_processes ${NUM_PROCESSES} \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m train.stage2_pfopsd \
    --config train/configs/pfopsd.yaml"

if [ -n "${CONDA_ENV}" ]; then
    conda run -n "${CONDA_ENV}" --no-capture-output ${_LAUNCH}
else
    eval ${_LAUNCH}
fi
