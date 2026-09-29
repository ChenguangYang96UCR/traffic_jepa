#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Cross-city transfer:
#   pretrain: all selected Alameda sensors except Fremont
#   fine-tune/validation/test: Fremont's released train/val/test splits
#
# Examples:
#   bash scripts/validation/run_AlamedaCities_pretrain_Fremont_finetune.sh all
#   ALAMEDA_CITIES='Oakland,Hayward' bash ... pretrain
#   bash scripts/validation/run_AlamedaCities_pretrain_Fremont_finetune.sh full

MODE="${1:-all}"
ALAMEDA_ROOT="${ALAMEDA_ROOT:-./dataset/Alameda/}"
FREMONT_ROOT="${FREMONT_ROOT:-./dataset/Fremont/}"
ALAMEDA_CITIES="${ALAMEDA_CITIES:-}"
ALAMEDA_EXCLUDE_CITIES="${ALAMEDA_EXCLUDE_CITIES:-Fremont}"
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

PRETRAIN_TAG="${PRETRAIN_TAG:-AlamedaOtherCities_joint_pretrain}"
ZERO_SHOT_TAG="${ZERO_SHOT_TAG:-AlamedaOtherCities_to_Fremont_zero_shot}"
FULL_TAG="${FULL_TAG:-AlamedaOtherCities_to_Fremont_full}"
PARTIAL_TAG="${PARTIAL_TAG:-AlamedaOtherCities_to_Fremont_partial_L${PARTIAL_LAYERS}}"
GRADUAL_TAG="${GRADUAL_TAG:-AlamedaOtherCities_to_Fremont_gradual_H${GRADUAL_HEAD_EPOCHS}_P${GRADUAL_PARTIAL_EPOCHS}_L${PARTIAL_LAYERS}}"
LORA_TAG="${LORA_TAG:-AlamedaOtherCities_to_Fremont_lora_R${LORA_RANK}_A${LORA_ALPHA}_D${LORA_DROPOUT}_LR${LORA_LR_SCALE}}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-./checkpoints/${PRETRAIN_TAG}_0/checkpoint.pth}"
export PRETRAIN_TAG ZERO_SHOT_TAG FULL_TAG PARTIAL_TAG GRADUAL_TAG LORA_TAG

MODEL_ARGS=(
  --seed 2026
  --model TopoJEPA
  --features M
  --freq t
  --seq_len 12
  --label_len 6
  --pred_len 12
  --d_model 128
  --n_heads 8
  --e_layers 2
  --d_ff 256
  --dropout 0.1
  --batch_size 32
  --num_workers 4
  --patience 5
  --learning_rate 0.0001
  --model_variant jepa
  --topo_weight 0
  --text_weight 0
)

ALAMEDA_ARGS=(
  "${MODEL_ARGS[@]}"
  --data AlamedaCities
  --root_path "$ALAMEDA_ROOT"
  --alameda_exclude_cities "$ALAMEDA_EXCLUDE_CITIES"
)
if [[ -n "$ALAMEDA_CITIES" ]]; then
  ALAMEDA_ARGS+=(--alameda_cities "$ALAMEDA_CITIES")
fi

FREMONT_ARGS=(
  "${MODEL_ARGS[@]}"
  --data Fremont
  --root_path "$FREMONT_ROOT"
  --enc_in 93
  --dec_in 93
  --c_out 93
)

require_checkpoint() {
  if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
    echo "Missing cross-city pretraining checkpoint: $PRETRAIN_CHECKPOINT" >&2
    echo "Run this script with 'pretrain' or 'all' first." >&2
    exit 1
  fi
}

run_pretrain() {
  python -u run.py "${ALAMEDA_ARGS[@]}" \
    --is_training 1 --model_id Alameda_other_cities_pretrain \
    --training_stage pretrain --experiment_tag "$PRETRAIN_TAG" \
    --train_epochs "$PRETRAIN_EPOCHS" --jepa_weight "$JEPA_WEIGHT" \
    --des other_cities_forecast_plus_jepa_pretrain
}

run_zero_shot() {
  require_checkpoint
  python -u run.py "${FREMONT_ARGS[@]}" \
    --is_training 0 --model_id Fremont_zero_shot \
    --training_stage pretrain --jepa_weight "$JEPA_WEIGHT" \
    --eval_checkpoint "$PRETRAIN_CHECKPOINT" --experiment_tag "$ZERO_SHOT_TAG"
}

run_finetune() {
  local strategy="$1"
  local tag="$2"
  shift 2
  require_checkpoint
  python -u run.py "${FREMONT_ARGS[@]}" \
    --is_training 1 --model_id "Fremont_${strategy}_finetune" \
    --training_stage finetune --finetune_strategy "$strategy" \
    --pretrained_checkpoint "$PRETRAIN_CHECKPOINT" \
    --experiment_tag "$tag" --train_epochs "$FINETUNE_EPOCHS" \
    --jepa_weight 0 --des other_cities_pretrain_Fremont_finetune "$@"
}

run_full() {
  run_finetune full "$FULL_TAG"
}

run_partial() {
  run_finetune partial "$PARTIAL_TAG" --partial_unfreeze_layers "$PARTIAL_LAYERS"
}

run_gradual() {
  run_finetune gradual "$GRADUAL_TAG" \
    --gradual_head_epochs "$GRADUAL_HEAD_EPOCHS" \
    --gradual_partial_epochs "$GRADUAL_PARTIAL_EPOCHS" \
    --partial_unfreeze_layers "$PARTIAL_LAYERS"
}

run_lora() {
  run_finetune lora "$LORA_TAG" \
    --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" --lora_lr_scale "$LORA_LR_SCALE"
}

print_summary() {
  python - <<'PY'
from pathlib import Path
import os
import numpy as np

runs = (
    ('Other-city pretrain -> Fremont zero-shot', os.environ['ZERO_SHOT_TAG']),
    ('Other-city pretrain + Fremont full', os.environ['FULL_TAG']),
    ('Other-city pretrain + Fremont partial', os.environ['PARTIAL_TAG']),
    ('Other-city pretrain + Fremont gradual', os.environ['GRADUAL_TAG']),
    ('Other-city pretrain + Fremont LoRA', os.environ['LORA_TAG']),
)
print('\nCross-city transfer evaluation (Fremont test set)')
print(f"{'Method':44s} {'MAE':>12s} {'MSE':>12s} {'RMSE':>12s}")
for name, tag in runs:
    path = Path('results') / f'{tag}_0' / 'metrics.npy'
    if not path.exists():
        print(f'{name:44s} {"missing":>38s}')
        continue
    mae, mse, rmse, _, _ = np.load(path)
    print(f'{name:44s} {mae:12.7f} {mse:12.7f} {rmse:12.7f}')
PY
}

case "$MODE" in
  pretrain) run_pretrain ;;
  zero-shot) run_zero_shot ;;
  full) run_full ;;
  partial) run_partial ;;
  gradual) run_gradual ;;
  lora) run_lora ;;
  summary) print_summary ;;
  all)
    run_pretrain
    run_zero_shot
    run_full
    run_partial
    run_gradual
    run_lora
    print_summary
    ;;
  *)
    echo "Usage: $0 {pretrain|zero-shot|full|partial|gradual|lora|summary|all}" >&2
    exit 2
    ;;
esac
