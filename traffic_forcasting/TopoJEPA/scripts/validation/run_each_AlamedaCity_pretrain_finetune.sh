#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# For every selected city, pretrain and fine-tune on that same city's sensors.
# Pretraining and fine-tuning both use Alameda's released train/val/test files;
# only the sensor axis changes between cities.

MODE="${1:-all}"
ALAMEDA_ROOT="${ALAMEDA_ROOT:-./dataset/Alameda/}"
SENSORS_FILE="${SENSORS_FILE:-sensors.csv}"
CITIES="${CITIES:-}"
FINETUNE_STRATEGY="${FINETUNE_STRATEGY:-full}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
JEPA_WEIGHT="${JEPA_WEIGHT:-0.1}"
PARTIAL_LAYERS="${PARTIAL_LAYERS:-1}"
GRADUAL_HEAD_EPOCHS="${GRADUAL_HEAD_EPOCHS:-2}"
GRADUAL_PARTIAL_EPOCHS="${GRADUAL_PARTIAL_EPOCHS:-3}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0}"
LORA_LR_SCALE="${LORA_LR_SCALE:-1}"

case "$FINETUNE_STRATEGY" in
  full|partial|gradual|lora) ;;
  *) echo 'FINETUNE_STRATEGY must be full, partial, gradual, or lora' >&2; exit 2 ;;
esac
case "$MODE" in
  all|pretrain|finetune|summary) ;;
  *) echo "Usage: $0 {all|pretrain|finetune|summary}" >&2; exit 2 ;;
esac

CITY_LIST=(python ../Preprocess/list_alameda_cities.py "$ALAMEDA_ROOT/$SENSORS_FILE")
if [[ -n "$CITIES" ]]; then
  CITY_LIST+=(--cities "$CITIES")
fi

MODEL_ARGS=(
  --seed 2026 --model TopoJEPA --data AlamedaCities
  --root_path "$ALAMEDA_ROOT" --alameda_sensors_file "$SENSORS_FILE"
  --alameda_exclude_cities '' --features M --freq t
  --seq_len 12 --label_len 6 --pred_len 12
  --d_model 128 --n_heads 8 --e_layers 2 --d_ff 256 --dropout 0.1
  --batch_size 32 --num_workers 4 --patience 5 --learning_rate 0.0001
  --model_variant jepa --topo_weight 0 --text_weight 0
)

finetune_extra=()
case "$FINETUNE_STRATEGY" in
  partial)
    finetune_extra=(--partial_unfreeze_layers "$PARTIAL_LAYERS") ;;
  gradual)
    finetune_extra=(--gradual_head_epochs "$GRADUAL_HEAD_EPOCHS"
      --gradual_partial_epochs "$GRADUAL_PARTIAL_EPOCHS"
      --partial_unfreeze_layers "$PARTIAL_LAYERS") ;;
  lora)
    finetune_extra=(--lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA"
      --lora_dropout "$LORA_DROPOUT" --lora_lr_scale "$LORA_LR_SCALE") ;;
esac

run_city_pretrain() {
  local city="$1" slug="$2" nodes="$3"
  local tag="City_${slug}_joint_pretrain"
  python -u run.py "${MODEL_ARGS[@]}" --alameda_cities "$city" \
    --enc_in "$nodes" --dec_in "$nodes" --c_out "$nodes" \
    --is_training 1 --model_id "${slug}_pretrain" \
    --training_stage pretrain --experiment_tag "$tag" \
    --train_epochs "$PRETRAIN_EPOCHS" --jepa_weight "$JEPA_WEIGHT" \
    --des same_city_forecast_plus_jepa_pretrain
}

run_city_finetune() {
  local city="$1" slug="$2" nodes="$3"
  local pretrain_tag="City_${slug}_joint_pretrain"
  local finetune_tag="City_${slug}_${FINETUNE_STRATEGY}_finetune"
  local checkpoint="./checkpoints/${pretrain_tag}_0/checkpoint.pth"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint for $city: $checkpoint" >&2
    echo "Run this script with 'pretrain' or 'all' first." >&2
    exit 1
  fi
  python -u run.py "${MODEL_ARGS[@]}" --alameda_cities "$city" \
    --enc_in "$nodes" --dec_in "$nodes" --c_out "$nodes" \
    --is_training 1 --model_id "${slug}_${FINETUNE_STRATEGY}_finetune" \
    --training_stage finetune --finetune_strategy "$FINETUNE_STRATEGY" \
    --pretrained_checkpoint "$checkpoint" --experiment_tag "$finetune_tag" \
    --train_epochs "$FINETUNE_EPOCHS" --jepa_weight 0 \
    --des same_city_pretrain_finetune "${finetune_extra[@]}"
}

print_city_summary() {
  printf '\nSame-city evaluation (each row uses that city test sensors)\n'
  printf '%-24s %7s %-12s %12s %12s %12s\n' City Sensors Stage MAE MSE RMSE
  while IFS=$'\t' read -r city slug nodes; do
    local pretrain_path="results/City_${slug}_joint_pretrain_0/metrics.npy"
    local finetune_path="results/City_${slug}_${FINETUNE_STRATEGY}_finetune_0/metrics.npy"
    python - "$city" "$nodes" "$pretrain_path" "$finetune_path" "$FINETUNE_STRATEGY" <<'PY'
import sys
from pathlib import Path
import numpy as np

city, nodes, pretrain_path, finetune_path, strategy = sys.argv[1:]
for stage, filename in [('pretrain', pretrain_path), (strategy, finetune_path)]:
    path = Path(filename)
    if path.exists():
        mae, mse, rmse, _, _ = np.load(path)
        print(f'{city[:24]:24s} {int(nodes):7d} {stage:12s} {mae:12.7f} {mse:12.7f} {rmse:12.7f}')
    else:
        print(f'{city[:24]:24s} {int(nodes):7d} {stage:12s} {"missing":>38s}')
PY
  done < <("${CITY_LIST[@]}")
}

if [[ "$MODE" == summary ]]; then
  print_city_summary
  exit 0
fi

while IFS=$'\t' read -r city slug nodes; do
  echo "===== City: $city | sensors: $nodes ====="
  if [[ "$MODE" == all || "$MODE" == pretrain ]]; then
    run_city_pretrain "$city" "$slug" "$nodes"
  fi
  if [[ "$MODE" == all || "$MODE" == finetune ]]; then
    run_city_finetune "$city" "$slug" "$nodes"
  fi
done < <("${CITY_LIST[@]}")

print_city_summary
