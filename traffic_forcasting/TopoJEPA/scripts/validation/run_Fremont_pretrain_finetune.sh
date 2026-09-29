#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Usage:
#   bash scripts/validation/run_Fremont_pretrain_finetune.sh pretrain
#   bash scripts/validation/run_Fremont_pretrain_finetune.sh full
#   bash scripts/validation/run_Fremont_pretrain_finetune.sh partial
#   bash scripts/validation/run_Fremont_pretrain_finetune.sh lora
#   bash scripts/validation/run_Fremont_pretrain_finetune.sh summary
#   bash scripts/validation/run_Fremont_pretrain_finetune.sh all
#
# `all` evaluates joint pretraining directly, then starts all fine-tuning modes
# adaptation from that same online forecasting checkpoint.

MODE="${1:-all}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
PARTIAL_LAYERS="${PARTIAL_LAYERS:-1}"
GRADUAL_HEAD_EPOCHS="${GRADUAL_HEAD_EPOCHS:-2}"
GRADUAL_PARTIAL_EPOCHS="${GRADUAL_PARTIAL_EPOCHS:-3}"
PRETRAIN_TAG="${PRETRAIN_TAG:-Fremont_joint_pretrain}"
JEPA_WEIGHT="${JEPA_WEIGHT:-0.1}"
PRETRAIN_OBJECTIVE="${PRETRAIN_OBJECTIVE:-forecast}"
PRETRAIN_MASK_STRATEGY="${PRETRAIN_MASK_STRATEGY:-none}"
PRETRAIN_MASK_RATIO="${PRETRAIN_MASK_RATIO:-0.25}"
RECONSTRUCTION_WEIGHT="${RECONSTRUCTION_WEIGHT:-0.1}"
FULL_TAG="${FULL_TAG:-Fremont_joint_pretrain_full}"
PARTIAL_TAG="${PARTIAL_TAG:-Fremont_joint_pretrain_partial_L${PARTIAL_LAYERS}}"
export PRETRAIN_TAG FULL_TAG PARTIAL_TAG
GRADUAL_TAG="${GRADUAL_TAG:-Fremont_joint_pretrain_gradual_H${GRADUAL_HEAD_EPOCHS}_P${GRADUAL_PARTIAL_EPOCHS}_L${PARTIAL_LAYERS}}"
export GRADUAL_TAG
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0}"
LORA_LR_SCALE="${LORA_LR_SCALE:-1}"
LORA_TAG="${LORA_TAG:-Fremont_joint_pretrain_lora_R${LORA_RANK}_A${LORA_ALPHA}_D${LORA_DROPOUT}_LR${LORA_LR_SCALE}}"
export LORA_TAG
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-./checkpoints/${PRETRAIN_TAG}_0/checkpoint.pth}"
ADAPTATION_ROOT="${ADAPTATION_ROOT:-}"
if [[ -n "$ADAPTATION_ROOT" ]]; then
  ADAPTATION_TAG="${ADAPTATION_TAG:-test622}"
  PRETRAIN_TAG="${PRETRAIN_TAG}_${ADAPTATION_TAG}"
  FULL_TAG="${FULL_TAG}_${ADAPTATION_TAG}"
  PARTIAL_TAG="${PARTIAL_TAG}_${ADAPTATION_TAG}"
  GRADUAL_TAG="${GRADUAL_TAG}_${ADAPTATION_TAG}"
  LORA_TAG="${LORA_TAG}_${ADAPTATION_TAG}"
fi

COMMON_ARGS=(
  --seed 2026
  --is_training 1
  --model TopoJEPA
  --data Fremont
  --root_path ./dataset/Fremont/
  --features M
  --freq t
  --seq_len 12
  --label_len 6
  --pred_len 12
  --enc_in 93
  --dec_in 93
  --c_out 93
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
  --pretrain_objective "$PRETRAIN_OBJECTIVE"
  --pretrain_mask_strategy "$PRETRAIN_MASK_STRATEGY"
  --pretrain_mask_ratio "$PRETRAIN_MASK_RATIO"
  --reconstruction_weight "$RECONSTRUCTION_WEIGHT"
  --topo_weight 0
  --text_weight 0
)
if [[ "${FORECAST_MASK:-0}" == "1" ]]; then
  COMMON_ARGS+=(
    --forecast_mask
    --forecast_mask_floor "${FORECAST_MASK_FLOOR:-0.1}"
    --forecast_mask_target "${FORECAST_MASK_TARGET:-0.5}"
    --forecast_mask_budget_weight "${FORECAST_MASK_BUDGET_WEIGHT:-0.1}"
    --forecast_mask_entropy_weight "${FORECAST_MASK_ENTROPY_WEIGHT:-0.01}"
    --forecast_mask_smooth_weight "${FORECAST_MASK_SMOOTH_WEIGHT:-0.01}"
  )
fi
if [[ -n "$ADAPTATION_ROOT" ]]; then
  COMMON_ARGS+=(--fremont_adaptation_root "$ADAPTATION_ROOT")
fi

run_baseline() {
  require_pretrain_checkpoint
  python -u run.py "${COMMON_ARGS[@]}" \
    --is_training 0 --model_id Fremont_pretrain_baseline \
    --training_stage pretrain --jepa_weight "$JEPA_WEIGHT" \
    --eval_checkpoint "$PRETRAIN_CHECKPOINT" --experiment_tag "$PRETRAIN_TAG"
}

run_pretrain() {
  python -u run.py "${COMMON_ARGS[@]}" \
    --model_id Fremont_pretrain \
    --training_stage pretrain \
    --experiment_tag "$PRETRAIN_TAG" \
    --train_epochs "$PRETRAIN_EPOCHS" \
    --jepa_weight "$JEPA_WEIGHT" \
    --des forecast_plus_jepa_pretrain
}

require_pretrain_checkpoint() {
  if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
    echo "Missing pretraining checkpoint: $PRETRAIN_CHECKPOINT" >&2
    echo "Run this script with 'pretrain' or 'all' first." >&2
    exit 1
  fi
}

run_full_finetune() {
  require_pretrain_checkpoint
  python -u run.py "${COMMON_ARGS[@]}" \
    --model_id Fremont_full_finetune \
    --training_stage finetune \
    --finetune_strategy full \
    --pretrained_checkpoint "$PRETRAIN_CHECKPOINT" \
    --experiment_tag "$FULL_TAG" \
    --train_epochs "$FINETUNE_EPOCHS" \
    --jepa_weight 0 \
    --des jepa_pretrain_full_finetune
}

run_partial_finetune() {
  require_pretrain_checkpoint
  python -u run.py "${COMMON_ARGS[@]}" \
    --model_id Fremont_partial_finetune \
    --training_stage finetune \
    --finetune_strategy partial \
    --partial_unfreeze_layers "$PARTIAL_LAYERS" \
    --pretrained_checkpoint "$PRETRAIN_CHECKPOINT" \
    --experiment_tag "$PARTIAL_TAG" \
    --train_epochs "$FINETUNE_EPOCHS" \
    --jepa_weight 0 \
    --des jepa_pretrain_partial_finetune
}

run_gradual_finetune() {
  require_pretrain_checkpoint
  python -u run.py "${COMMON_ARGS[@]}" \
    --model_id Fremont_gradual_finetune \
    --training_stage finetune \
    --finetune_strategy gradual \
    --gradual_head_epochs "$GRADUAL_HEAD_EPOCHS" \
    --gradual_partial_epochs "$GRADUAL_PARTIAL_EPOCHS" \
    --partial_unfreeze_layers "$PARTIAL_LAYERS" \
    --pretrained_checkpoint "$PRETRAIN_CHECKPOINT" \
    --experiment_tag "$GRADUAL_TAG" \
    --train_epochs "$FINETUNE_EPOCHS" \
    --jepa_weight 0 \
    --des joint_pretrain_gradual_finetune
}

run_lora_finetune() {
  require_pretrain_checkpoint
  python -u run.py "${COMMON_ARGS[@]}" \
    --model_id Fremont_lora_finetune \
    --training_stage finetune \
    --finetune_strategy lora \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_lr_scale "$LORA_LR_SCALE" \
    --pretrained_checkpoint "$PRETRAIN_CHECKPOINT" \
    --experiment_tag "$LORA_TAG" \
    --train_epochs "$FINETUNE_EPOCHS" \
    --jepa_weight 0 \
    --des joint_pretrain_lora_finetune
}

print_summary() {
  python - <<'PY'
from pathlib import Path
import os
import numpy as np

runs = (
    ('Joint pretrain only (direct test)',
     Path('results') / (os.environ['PRETRAIN_TAG'] + '_0') / 'metrics.npy'),
    ('Pretrain + full fine-tune',
     Path('results') / (os.environ['FULL_TAG'] + '_0') / 'metrics.npy'),
    ('Pretrain + partial fine-tune',
     Path('results') / (os.environ['PARTIAL_TAG'] + '_0') / 'metrics.npy'),
    ('Pretrain + gradual fine-tune',
     Path('results') / (os.environ['GRADUAL_TAG'] + '_0') / 'metrics.npy'),
    ('Pretrain + LoRA fine-tune',
     Path('results') / (os.environ['LORA_TAG'] + '_0') / 'metrics.npy'),
)
print('\nTransfer evaluation summary')
print(f"{'Method':36s} {'MAE':>12s} {'MSE':>12s} {'RMSE':>12s}")
for name, path in runs:
    if not path.exists():
        print(f'{name:36s} {"missing":>38s}')
        continue
    mae, mse, rmse, _, _ = np.load(path)
    print(f'{name:36s} {mae:12.7f} {mse:12.7f} {rmse:12.7f}')
PY
}

case "$MODE" in
  transfer)
    if [[ -z "$ADAPTATION_ROOT" ]]; then
      echo 'transfer requires ADAPTATION_ROOT' >&2
      exit 2
    fi
    run_baseline
    run_full_finetune
    run_partial_finetune
    run_gradual_finetune
    run_lora_finetune
    print_summary
    ;;
  baseline)
    run_baseline
    ;;
  pretrain)
    run_pretrain
    ;;
  full)
    run_full_finetune
    ;;
  partial)
    run_partial_finetune
    ;;
  summary)
    print_summary
    ;;
  gradual)
    run_gradual_finetune
    print_summary
    ;;
  all)
    # Ensure all adaptations use the checkpoint produced by this invocation.
    PRETRAIN_CHECKPOINT="./checkpoints/${PRETRAIN_TAG}_0/checkpoint.pth"
    run_pretrain
    run_full_finetune
    run_partial_finetune
    run_gradual_finetune
    run_lora_finetune
    print_summary
    ;;
  lora)
    run_lora_finetune
    print_summary
    ;;
  *)
    echo "Usage: $0 {pretrain|full|partial|gradual|lora|baseline|transfer|summary|all}" >&2
    exit 2
    ;;
esac
