#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-dataset/Fremont}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/fremont_random_k9}"

python train.py \
  --data "$DATA_ROOT" \
  --data-format incident \
  --file-pattern 'incident_{split}.npy' \
  --traffic-feature 0 \
  --graph-mode file \
  --adjacency "$DATA_ROOT/adj_matrix.npy" \
  --window 12 \
  --mask-mode random \
  --mask-k 9 \
  --dim 128 \
  --heads 4 \
  --layers 2 \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --finetune-strategy full \
  --jepa-weight 0.1 \
  --batch-size 32 \
  --lr 0.0001 \
  --output "$OUTPUT_DIR"

