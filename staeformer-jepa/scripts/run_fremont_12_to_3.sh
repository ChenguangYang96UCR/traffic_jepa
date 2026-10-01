#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-../TopoJEPA/dataset/Fremont}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/fremont_12_to_3}"
SEED="${SEED:-2026}"

python run.py \
  --mode compare \
  --data "$DATA_ROOT" \
  --file-pattern 'incident_{split}.npy' \
  --traffic-feature 0 \
  --input-steps 12 \
  --pred-steps 3 \
  --steps-per-day 288 \
  --input-embedding-dim 24 \
  --tod-embedding-dim 24 \
  --dow-embedding-dim 24 \
  --adaptive-embedding-dim 80 \
  --feed-forward-dim 256 \
  --heads 4 \
  --layers 3 \
  --predictor-layers 2 \
  --pretrain-epochs 20 \
  --finetune-epochs 50 \
  --pretrain-lr 0.0001 \
  --finetune-lr 0.001 \
  --forecast-loss mae \
  --batch-size 16 \
  --patience 10 \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"

