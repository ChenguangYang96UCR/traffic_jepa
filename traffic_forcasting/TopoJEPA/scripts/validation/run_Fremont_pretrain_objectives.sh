#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Compare auxiliary pretraining objectives while keeping JEPA, the backbone,
# source data and full fine-tuning protocol fixed.
#
# Usage:
#   bash scripts/validation/run_Fremont_pretrain_objectives.sh all
#   bash scripts/validation/run_Fremont_pretrain_objectives.sh forecast
#   bash scripts/validation/run_Fremont_pretrain_objectives.sh summary

MODE="${1:-all}"
OBJECTIVES=(jepa forecast masked_forecast reconstruction forecast_reconstruction)
MASK_STRATEGY="${MASK_STRATEGY:-temporal}"
MASK_RATIO="${MASK_RATIO:-0.25}"
export MASK_STRATEGY MASK_RATIO
BASE_RUNNER="scripts/validation/run_Fremont_pretrain_finetune.sh"

configure_objective() {
  local objective="$1"
  PRETRAIN_OBJECTIVE="$objective"
  PRETRAIN_MASK_STRATEGY="none"
  if [[ "$objective" == "masked_forecast" ||
        "$objective" == "reconstruction" ||
        "$objective" == "forecast_reconstruction" ]]; then
    PRETRAIN_MASK_STRATEGY="$MASK_STRATEGY"
  fi
  PRETRAIN_TAG="Fremont_pre_${objective}_${PRETRAIN_MASK_STRATEGY}_r${MASK_RATIO}"
  FULL_TAG="Fremont_pre_${objective}_${PRETRAIN_MASK_STRATEGY}_r${MASK_RATIO}_full"
  PRETRAIN_CHECKPOINT="./checkpoints/${PRETRAIN_TAG}_0/checkpoint.pth"
  export PRETRAIN_OBJECTIVE PRETRAIN_MASK_STRATEGY PRETRAIN_TAG FULL_TAG
  export PRETRAIN_CHECKPOINT
  export PRETRAIN_MASK_RATIO="$MASK_RATIO"
}

run_objective() {
  local objective="$1"
  configure_objective "$objective"
  echo "=== Pretrain objective: $objective; mask: $PRETRAIN_MASK_STRATEGY ==="
  bash "$BASE_RUNNER" pretrain
  bash "$BASE_RUNNER" full
}

print_summary() {
  python - <<'PY'
from pathlib import Path
import os
import numpy as np

masked = os.environ.get('MASK_STRATEGY', 'temporal')
ratio = os.environ.get('MASK_RATIO', '0.25')
specs = (
    ('jepa', 'none', False),
    ('forecast', 'none', True),
    ('masked_forecast', masked, True),
    ('reconstruction', masked, False),
    ('forecast_reconstruction', masked, True),
)
print('\nJEPA-fixed pretraining objective comparison (Fremont test set)')
print(f"{'Objective':31s} {'Direct MAE':>12s} {'FT MAE':>12s} "
      f"{'FT MSE':>12s} {'FT RMSE':>12s}")
for objective, default_mask, direct_valid in specs:
    mask = default_mask
    pre = Path('results') / f'Fremont_pre_{objective}_{mask}_r{ratio}_0' / 'metrics.npy'
    fine = Path('results') / f'Fremont_pre_{objective}_{mask}_r{ratio}_full_0' / 'metrics.npy'
    direct = 'n/a'
    if direct_valid and pre.exists():
        direct = f'{np.load(pre)[0]:.7f}'
    if fine.exists():
        mae, mse, rmse, _, _ = np.load(fine)
        print(f'{objective:31s} {direct:>12s} {mae:12.7f} '
              f'{mse:12.7f} {rmse:12.7f}')
    else:
        print(f'{objective:31s} {direct:>12s} {"missing":>38s}')
print('\nDirect metrics are n/a when pretraining never trained the forecast head.')
PY
}

case "$MODE" in
  all)
    for objective in "${OBJECTIVES[@]}"; do
      run_objective "$objective"
    done
    print_summary
    ;;
  summary)
    print_summary
    ;;
  jepa|forecast|masked_forecast|reconstruction|forecast_reconstruction)
    run_objective "$MODE"
    print_summary
    ;;
  *)
    echo "Usage: $0 {all|summary|jepa|forecast|masked_forecast|reconstruction|forecast_reconstruction}" >&2
    exit 2
    ;;
esac
