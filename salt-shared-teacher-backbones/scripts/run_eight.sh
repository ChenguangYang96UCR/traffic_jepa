#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

read -r -a MODEL_LIST <<< "${BACKBONES:-pdformer flashst patchstg testam patchtst staeformer stgormer tsformer}"
read -r -a SEED_LIST <<< "${SEEDS:-2024 2025 2026}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/shared_teacher_backbones}"
export OUTPUT_ROOT
HORIZON="${HORIZON:-12}"
if [[ "${#SEED_LIST[@]}" -ne 3 && -z "${ALLOW_NONTHREE_SEEDS:-}" ]]; then
  echo "ERROR: exactly three seeds are required; got: ${SEED_LIST[*]}" >&2
  echo "Set ALLOW_NONTHREE_SEEDS=1 only for smoke tests." >&2
  exit 2
fi

for seed in "${SEED_LIST[@]}"; do
  if [[ -n "${TEACHER_CHECKPOINT:-}" ]]; then
    teacher="$TEACHER_CHECKPOINT"
  else
    teacher="${SHARED_TEACHER_ROOT:-$OUTPUT_ROOT/shared_teacher}/horizon_$HORIZON/seed_$seed/best_teacher.pt"
  fi
  if [[ ! -f "$teacher" ]]; then
    if [[ -n "${TEACHER_CHECKPOINT:-}" ]]; then
      echo "ERROR: TEACHER_CHECKPOINT not found: $teacher" >&2
      exit 2
    fi
    echo "===== seed=$seed shared Teacher ====="
    SEED="$seed" HORIZON="$HORIZON" bash scripts/train_shared_teacher.sh
  else
    echo "===== reusing shared Teacher: $teacher ====="
  fi
  for model in "${MODEL_LIST[@]}"; do
    echo "===== seed=$seed model=$model ====="
    SEED="$seed" HORIZON="$HORIZON" MODEL="$model" bash scripts/run_one.sh
  done
done

python summarize.py \
  --root "$OUTPUT_ROOT" \
  --expected-seeds "${EXPECTED_SEEDS:-${#SEED_LIST[@]}}"
