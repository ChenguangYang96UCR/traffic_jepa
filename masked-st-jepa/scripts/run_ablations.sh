#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-dataset/Fremont}"

for MODE in random persistent spatial_block; do
  for K in 5 9 19 28; do
    OUTPUT_DIR="runs/${MODE}_k${K}" \
    python train.py \
      --data "$DATA_ROOT" \
      --data-format incident \
      --adjacency "$DATA_ROOT/adj_matrix.npy" \
      --mask-mode "$MODE" \
      --mask-k "$K" \
      --pretrain-epochs 20 \
      --finetune-epochs 20 \
      --output "runs/${MODE}_k${K}"
  done
done

