#!/bin/bash
# run_sft.sh — Stage 1 Protocol SFT
#
# Usage:
#   # 4-GPU training (recommended):
#   bash train/scripts/run_sft.sh
#
#   # Custom number of GPUs:
#   bash train/scripts/run_sft.sh 8
#
#   # Single-GPU:
#   bash train/scripts/run_sft.sh 1

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ── Environment ──────────────────────────────────────────────────────────────
# Set CONDA_ENV to your environment name, or leave empty to use the current env.
CONDA_ENV="${CONDA_ENV:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ── Accelerate args ───────────────────────────────────────────────────────────
NUM_PROCESSES="${1:-4}"    # override with first argument

echo "======================================================"
echo "  PF-OPSD Stage 1 — Protocol SFT"
echo "  Model : $(grep model_name_or_path train/configs/sft.yaml)"
echo "  GPUs  : ${NUM_PROCESSES}"
echo "  Root  : ${REPO_ROOT}"
echo "======================================================"

_LAUNCH="accelerate launch \
    --num_processes ${NUM_PROCESSES} \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m train.stage1_sft \
    --config train/configs/sft.yaml"

if [ -n "${CONDA_ENV}" ]; then
    conda run -n "${CONDA_ENV}" --no-capture-output ${_LAUNCH}
else
    eval ${_LAUNCH}
fi
