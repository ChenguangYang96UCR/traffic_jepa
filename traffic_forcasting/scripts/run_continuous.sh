#!/usr/bin/env bash
set -euo pipefail

DATA_FILE="${DATA_FILE:?Set DATA_FILE to a [time,node] .csv/.npy/.npz file}"
ADJACENCY="${ADJACENCY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/continuous_random_k9}"

GRAPH_ARGS=(--graph-mode correlation --graph-top-k 8)
if [[ -n "$ADJACENCY" ]]; then
  GRAPH_ARGS=(--graph-mode file --adjacency "$ADJACENCY")
fi

python train.py \
  --data "$DATA_FILE" \
  --data-format continuous \
  --window 12 \
  --stride 1 \
  --split-ratios 0.7 0.1 0.2 \
  "${GRAPH_ARGS[@]}" \
  --mask-mode random \
  --mask-k 9 \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --output "$OUTPUT_DIR"

