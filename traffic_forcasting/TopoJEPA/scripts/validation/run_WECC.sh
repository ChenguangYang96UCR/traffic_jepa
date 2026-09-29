#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

VARIANT="${VARIANT:-jepa}"
JEPA_WEIGHT="${JEPA_WEIGHT:-0.1}"
TOPO_WEIGHT="${TOPO_WEIGHT:-0.5}"
TEXT_WEIGHT="${TEXT_WEIGHT:-0.03}"
USE_GNN="${USE_GNN:-1}"
ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-0.01}"
TEXT_EMBEDDING_DIR="${TEXT_EMBEDDING_DIR:-./text_embedding_cache}"
ADJ_PATH="${ADJ_PATH:-./dataset/WECC/adjacent.csv}"
SEEDS="${SEEDS:-2022 2023 2024 2025 2026}"

GNN_ARGS=()
if [[ "$USE_GNN" == "1" ]]; then
  GNN_ARGS=(--use_gnn --adj_path "$ADJ_PATH" --gnn_layers 2 --gnn_dropout 0.1)
fi

for SEED in $SEEDS; do
  echo "===== Running WECC with seed=${SEED} ====="

  python -u run.py \
    --seed "$SEED" \
    --is_training 1 --model TopoJEPA --data custom --features M \
    --root_path ./dataset/WECC/ --data_path WECC.csv --target WAUW --freq h \
    --model_id WECC_96_96 --seq_len 96 --label_len 48 --pred_len 96 \
    --enc_in 28 --dec_in 28 --c_out 28 --d_model 128 --d_ff 256 \
    --e_layers 2 --n_heads 8 --batch_size 32 --learning_rate 0.0003 \
    --train_epochs 10 --patience 3 --log_interval 50 --num_workers 0 --itr 1 --des validation \
    --model_variant "$VARIANT" --jepa_weight "$JEPA_WEIGHT" --topo_weight "$TOPO_WEIGHT" \
    --text_weight "$TEXT_WEIGHT" --text_embedding_dir "$TEXT_EMBEDDING_DIR" \
    --alignment_weight "$ALIGNMENT_WEIGHT" --ema_momentum 0.996 "${GNN_ARGS[@]}"
done
