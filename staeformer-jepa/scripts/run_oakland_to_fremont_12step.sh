#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

SOURCE_ROOT="${SOURCE_ROOT:-../TopoJEPA/dataset/Oakland}"
TARGET_ROOT="${TARGET_ROOT:-../TopoJEPA/dataset/Fremont}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/oakland_to_fremont_12step}"
SEED="${SEED:-2026}"

python run_transfer.py \
  --mode compare \
  --source-data "$SOURCE_ROOT" \
  --target-data "$TARGET_ROOT" \
  --transfer-branch target \
  --pretrain-mask-mode future \
  --file-pattern 'incident_{split}.npy' \
  --traffic-feature 0 \
  --input-steps 12 \
  --pred-steps 12 \
  --masked-steps 12 \
  --input-embedding-dim 24 \
  --step-embedding-dim 24 \
  --sensor-embedding-dim 80 \
  --feed-forward-dim 256 \
  --heads 4 \
  --layers 3 \
  --predictor-layers 2 \
  --pretrain-epochs 20 \
  --finetune-epochs 50 \
  --pretrain-lr 0.0001 \
  --finetune-lr 0.001 \
  --encoder-lr-scale 0.1 \
  --forecast-loss mae \
  --batch-size 16 \
  --patience 10 \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"
