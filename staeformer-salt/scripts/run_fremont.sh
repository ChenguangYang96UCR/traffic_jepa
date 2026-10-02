#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-../dataset/Fremont}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/fremont_salt}"
SEED="${SEED:-2026}"

# No cross-city transfer: teacher, student, and downstream forecast training all
# use Fremont. Dataset classes still keep train/val/test strictly separated.
python run.py \
  --pipeline same_city \
  --pretrain-data "$DATA_ROOT" \
  --target-data "$DATA_ROOT" \
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
