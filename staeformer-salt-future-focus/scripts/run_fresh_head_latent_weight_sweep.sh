#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
SWEEP_ROOT="${SWEEP_ROOT:-runs/oakland_teacher128_berkeley_sweep}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$SWEEP_ROOT/teachers/blocks=8_ratio=0.5/teacher/best_teacher.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/fresh_head_latent_weight_sweep}"
LATENT_WEIGHTS="${LATENT_WEIGHTS:-0 0.01 0.03 0.1 0.3 1.0 3.0}"
SEED="${SEED:-2026}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-50}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"

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

for latent_weight in $LATENT_WEIGHTS; do
  run_dir="$OUTPUT_ROOT/lambda=$latent_weight"
  echo "=== Fresh-head run: latent_weight=$latent_weight ==="
  python run_abc_ablation.py \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --berkeley-data "$BERKELEY_ROOT" \
    --student-epochs "$STUDENT_EPOCHS" \
    --student-lr 0.0001 \
    --probe-lr 0.001 \
    --latent-weight "$latent_weight" \
    --c-head-init fresh \
    --only-c \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --finetune-lr 0.001 \
    --encoder-lr-scale 0.1 \
    --batch-size 16 \
    --patience 10 \
    --seed "$SEED" \
    --output "$run_dir"
done

python summarize_fresh_head_sweep.py --root "$OUTPUT_ROOT"
