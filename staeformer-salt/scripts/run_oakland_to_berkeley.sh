#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OAKLAND_ROOT="${OAKLAND_ROOT:-../TopoJEPA/dataset/Oakland}"
BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-runs/oakland_teacher/teacher/best_teacher.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/exp3_oakland_oakland_berkeley}"
SEED="${SEED:-2026}"

for root in "$OAKLAND_ROOT" "$BERKELEY_ROOT"; do
  for split in train val test; do
    file="$root/incident_${split}.npy"
    if [[ ! -f "$file" ]]; then
      echo "ERROR: dataset split not found: $file" >&2
      exit 2
    fi
  done
done

if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "ERROR: Oakland Teacher checkpoint not found: $TEACHER_CHECKPOINT" >&2
  exit 2
fi

python run.py \
  --pipeline cross_city \
  --mode student_downstream \
  --experiment-label "Oakland Teacher -> Oakland Student -> Berkeley" \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --student-data "$OAKLAND_ROOT" \
  --target-data "$BERKELEY_ROOT" \
  --input-steps 12 \
  --pred-steps 12 \
  --student-epochs 50 \
  --finetune-epochs 50 \
  --student-lr 0.0001 \
  --finetune-lr 0.001 \
  --encoder-lr-scale 0.1 \
  --batch-size 16 \
  --patience 10 \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"
