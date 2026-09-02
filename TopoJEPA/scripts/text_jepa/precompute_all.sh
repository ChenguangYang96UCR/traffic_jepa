#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

CACHE_DIR="${CACHE_DIR:-./text_embedding_cache}"
DEVICE="${DEVICE:-cuda}"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/WECC/ --data_path WECC.csv --target WAUW --freq h \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/Homeless/ --data_path Homeless.csv --target node_46 --freq w \
  --seq_len 16 --label_len 8 --pred_len 16 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/Illegal_Dumping/ --data_path Illegal_Dumping.csv --target node_46 --freq w \
  --seq_len 16 --label_len 8 --pred_len 16 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/Dangerous_Building_Complaint/ --data_path Dangerous_Building_Complaint.csv --target node_46 --freq w \
  --seq_len 16 --label_len 8 --pred_len 16 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/Opioid_Response_Unit/ --data_path Opioid_Response_Unit.csv --target node_46 --freq w \
  --seq_len 16 --label_len 8 --pred_len 16 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/BJ/ --data_path beijing.csv --target 1036 --freq h \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/Electricity_Price/ --data_path Electricity_Price.csv --target SK-price --freq 15min \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"


python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/influenza/ --data_path influenza_us.csv --target Wyoming --freq w \
  --seq_len 16 --label_len 8 --pred_len 16 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"


python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/nyiso/ --data_path nyiso.csv --target Series_11 --freq h \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/caiso/ --data_path caiso.csv --target Series_9 --freq h \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/chickenpox/ --data_path chickenpox.csv --target ZALA --freq w \
  --seq_len 16 --label_len 8 --pred_len 16 \
  --cache_dir "$CACHE_DIR" --device "$DEVICE"

python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/tuberculosis/ --data_path tuberculosis.csv --target 47_TB_OKINAWA --freq m \
  --seq_len 12 --label_len 6 --pred_len 12 \
  --cache_dir ./text_embedding_cache --device cuda