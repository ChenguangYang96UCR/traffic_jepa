#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

FREMONT_ROOT="${FREMONT_ROOT:-../dataset/Fremont}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/exp1_fremont_teacher_fremont_student}"
SEED="${SEED:-2026}"

python run.py \
  --pipeline fremont_direct \
  --mode all \
  --teacher-data "$FREMONT_ROOT" \
  --student-data "$FREMONT_ROOT" \
  --target-data "$FREMONT_ROOT" \
  --teacher-sensor-dim 0 \
  --input-steps 12 \
  --pred-steps 12 \
  --teacher-epochs 20 \
  --student-epochs 50 \
  --finetune-epochs 50 \
  --teacher-lr 0.001 \
  --student-lr 0.0001 \
  --finetune-lr 0.001 \
  --encoder-lr-scale 0.1 \
  --batch-size 16 \
  --patience 10 \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"
