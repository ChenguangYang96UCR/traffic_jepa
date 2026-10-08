#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

read -r -a MODEL_LIST <<< "${BACKBONES:-pdformer flashst patchstg testam patchtst staeformer stgormer tsformer}"
read -r -a HORIZON_LIST <<< "${HORIZONS:-6 9 12}"
SEED="${SEED:-2026}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/shared_teacher_horizon_sweep_seed2026}"
export OUTPUT_ROOT
export MASK_BLOCKS="${MASK_BLOCKS:-8}"
export TEACHER_FUTURE_BLOCK_RATIO="${TEACHER_FUTURE_BLOCK_RATIO:-0.5}"

if [[ "$SEED" -ne 2026 ]]; then
  echo "ERROR: the controlled horizon sweep is fixed to seed 2026; got $SEED" >&2
  exit 2
fi

for horizon in "${HORIZON_LIST[@]}"; do
  if [[ "$horizon" != 6 && "$horizon" != 9 && "$horizon" != 12 ]]; then
    echo "ERROR: supported horizons are 6, 9, and 12; got $horizon" >&2
    exit 2
  fi
  if [[ -n "${TEACHER_CHECKPOINT:-}" ]]; then
    teacher="$TEACHER_CHECKPOINT"
  else
    teacher="${SHARED_TEACHER_ROOT:-$OUTPUT_ROOT/shared_teacher}/horizon_$horizon/seed_$SEED/best_teacher.pt"
  fi
  if [[ ! -f "$teacher" ]]; then
    if [[ -n "${TEACHER_CHECKPOINT:-}" ]]; then
      echo "ERROR: TEACHER_CHECKPOINT not found: $teacher" >&2
      exit 2
    fi
    echo "===== seed=$SEED horizon=$horizon shared Teacher ====="
    HORIZON="$horizon" SEED="$SEED" bash scripts/train_shared_teacher.sh
  else
    echo "===== reusing shared Teacher: $teacher ====="
  fi
  for model in "${MODEL_LIST[@]}"; do
    echo "===== seed=$SEED horizon=$horizon model=$model ====="
    HORIZON_SWEEP=1 HORIZON="$horizon" SEED="$SEED" MODEL="$model" \
      bash scripts/run_one.sh
  done
done

python summarize_horizons.py \
  --root "$OUTPUT_ROOT" \
  --seed "$SEED"
