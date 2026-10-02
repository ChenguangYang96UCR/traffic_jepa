#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OAKLAND_ROOT="${OAKLAND_ROOT:-../dataset/Oakland}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/oakland_teacher}"
SEED="${SEED:-2026}"

# The shared teacher deliberately has no city-specific sensor embedding, so its
# frozen checkpoint can generate targets for both Oakland and Fremont students.
python run.py \
  --mode teacher \
  --teacher-data "$OAKLAND_ROOT" \
  --teacher-sensor-dim 0 \
  --input-steps 12 \
  --pred-steps 12 \
  --teacher-epochs 20 \
  --teacher-lr 0.001 \
  --batch-size 16 \
  --patience 10 \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"
