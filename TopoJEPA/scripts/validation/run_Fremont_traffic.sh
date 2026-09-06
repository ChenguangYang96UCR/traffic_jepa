#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Traffic-only JEPA: no incident metadata, temporal input, topology loss,
# graph encoder, adjacency matrix, text target, or alignment branch.
VARIANT="${VARIANT:-jepa}"
JEPA_WEIGHT="${JEPA_WEIGHT:-0.1}"

python -u run.py \
  --is_training 1 \
  --model_id Fremont_traffic \
  --model TopoJEPA \
  --data Fremont \
  --root_path ./dataset/Fremont/ \
  --features M \
  --freq t \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --enc_in 93 \
  --dec_in 93 \
  --c_out 93 \
  --d_model 128 \
  --n_heads 8 \
  --e_layers 2 \
  --d_ff 256 \
  --batch_size 32 \
  --num_workers 4 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.0001 \
  --model_variant "$VARIANT" \
  --jepa_weight "$JEPA_WEIGHT" \
  --text_weight 0 \
  --des traffic_only_no_topology
