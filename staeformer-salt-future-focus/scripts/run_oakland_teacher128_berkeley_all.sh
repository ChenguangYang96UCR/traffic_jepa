#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OAKLAND_ROOT="${OAKLAND_ROOT:-../TopoJEPA/dataset/Oakland}"
BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/oakland_teacher128_berkeley_all}"
BASELINE_SALT_ROOT="${BASELINE_SALT_ROOT:-runs/future_focus_berkeley}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$OUTPUT_DIR/teacher_oakland_d128/teacher/best_teacher.pt}"
SEED="${SEED:-2026}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-50}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"

for root in "$OAKLAND_ROOT" "$BERKELEY_ROOT"; do
  for split in train val test; do
    file="$root/incident_${split}.npy"
    if [[ ! -f "$file" ]]; then
      echo "ERROR: dataset split not found: $file" >&2
      exit 2
    fi
  done
done

# A node-agnostic 128-d Teacher: 64 traffic + 64 relative-step + 0 sensor.
# Four mask blocks allocate exactly three block draws to future and one to history.
if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
  python run.py \
    --mode teacher \
    --teacher-data "$OAKLAND_ROOT" \
    --teacher-input-dim 64 \
    --teacher-step-dim 64 \
    --teacher-sensor-dim 0 \
    --teacher-ff-dim 256 \
    --teacher-heads 4 \
    --teacher-layers 3 \
    --teacher-mask-policy future_biased \
    --teacher-future-block-ratio 0.75 \
    --mask-blocks 4 \
    --input-steps 12 \
    --pred-steps 12 \
    --teacher-epochs "$TEACHER_EPOCHS" \
    --teacher-lr 0.001 \
    --batch-size 16 \
    --patience 10 \
    --seed "$SEED" \
    --output "$OUTPUT_DIR/teacher_oakland_d128"
fi

if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "ERROR: d=128 Teacher checkpoint not found: $TEACHER_CHECKPOINT" >&2
  exit 2
fi

python run.py \
  --pipeline in_domain \
  --mode student_downstream \
  --experiment-label "Oakland Teacher d=128 -> Berkeley Student | all-step loss" \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --require-future-focused-teacher \
  --expected-teacher-dim 128 \
  --student-data "$BERKELEY_ROOT" \
  --target-data "$BERKELEY_ROOT" \
  --student-distill-scope all \
  --downstream-strategies frozen,full \
  --input-steps 12 \
  --pred-steps 12 \
  --student-epochs "$STUDENT_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --student-lr 0.0001 \
  --finetune-lr 0.001 \
  --encoder-lr-scale 0.1 \
  --batch-size 16 \
  --patience 10 \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"

BASELINE_RESULTS="$BASELINE_SALT_ROOT/exp2_oakland_berkeley/all/all_results.json"
if [[ -f "$BASELINE_RESULTS" ]]; then
  python compare_teacher_width.py \
    --baseline-root "$BASELINE_SALT_ROOT" \
    --teacher128-root "$OUTPUT_DIR"
else
  echo "d=32 baseline not found under $BASELINE_SALT_ROOT; skipping width comparison."
  echo "The d=128 result is available at $OUTPUT_DIR/summary.txt."
fi
