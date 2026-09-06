#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Incident-aware JEPA. Traffic is still the only prediction target; accident
# attributes and per-sensor distance features are optional conditioning data.
python -u run.py \
  --seed 2026 \
  --is_training 1 \
  --model_id Fremont_incident \
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
  --dropout 0.1 \
  --batch_size 32 \
  --num_workers 4 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.0001 \
  --model_variant jepa \
  --jepa_weight 0.1 \
  --text_weight 0 \
  --incident \
  --incident_sigma 4.0 \
  --incident_scale 1.0 \
  --des traffic_plus_incident_no_topology
