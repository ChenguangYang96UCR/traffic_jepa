#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export FORECAST_MASK=1
export FORECAST_MASK_FLOOR="${FORECAST_MASK_FLOOR:-0.1}"
export FORECAST_MASK_TARGET="${FORECAST_MASK_TARGET:-0.5}"
export FORECAST_MASK_BUDGET_WEIGHT="${FORECAST_MASK_BUDGET_WEIGHT:-0.1}"
export FORECAST_MASK_ENTROPY_WEIGHT="${FORECAST_MASK_ENTROPY_WEIGHT:-0.01}"
export FORECAST_MASK_SMOOTH_WEIGHT="${FORECAST_MASK_SMOOTH_WEIGHT:-0.01}"

# Separate names prevent masked experiments from overwriting existing runs.
export PRETRAIN_TAG="${PRETRAIN_TAG:-Fremont_masked_pretrain}"
export FULL_TAG="${FULL_TAG:-Fremont_masked_pretrain_full}"
export PARTIAL_TAG="${PARTIAL_TAG:-Fremont_masked_pretrain_partial_L${PARTIAL_LAYERS:-1}}"
export GRADUAL_TAG="${GRADUAL_TAG:-Fremont_masked_pretrain_gradual}"
export LORA_TAG="${LORA_TAG:-Fremont_masked_pretrain_lora}"
export PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-./checkpoints/${PRETRAIN_TAG}_0/checkpoint.pth}"

exec bash "$SCRIPT_DIR/run_Fremont_pretrain_finetune.sh" "${1:-all}"
