#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
SWEEP_ROOT="${SWEEP_ROOT:-runs/oakland_teacher128_berkeley_sweep}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$SWEEP_ROOT/teachers/blocks=8_ratio=0.5/teacher/best_teacher.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/best_salt_abc}"
SEED="${SEED:-2026}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-50}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"

for split in train val test; do
  file="$BERKELEY_ROOT/incident_${split}.npy"
  if [[ ! -f "$file" ]]; then
    echo "ERROR: Berkeley split not found: $file" >&2
    exit 2
  fi
done
if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "ERROR: best Teacher checkpoint not found: $TEACHER_CHECKPOINT" >&2
  exit 2
fi

python run_abc_ablation.py \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --berkeley-data "$BERKELEY_ROOT" \
  --student-epochs "$STUDENT_EPOCHS" \
  --student-lr 0.0001 \
  --probe-lr 0.001 \
  --latent-weight "$LATENT_WEIGHT" \
  --c-head-init warm \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --finetune-lr 0.001 \
  --encoder-lr-scale 0.1 \
  --batch-size 16 \
  --patience 10 \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"
