#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

SOURCE_ROOT="${SOURCE_ROOT:-../dataset/Oakland}"
TARGET_ROOT="${TARGET_ROOT:-../dataset/Fremont}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-runs/oakland_teacher/teacher/best_teacher.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/oakland_to_fremont_salt}"
SEED="${SEED:-2026}"

python run.py \
  --pipeline cross_city \
  --mode student_downstream \
  --experiment-label "Oakland Teacher -> Oakland Student -> Fremont" \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --student-data "$SOURCE_ROOT" \
  --target-data "$TARGET_ROOT" \
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
