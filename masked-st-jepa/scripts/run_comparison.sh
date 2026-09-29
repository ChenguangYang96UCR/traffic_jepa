#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-dataset/Fremont}"
RUN_ROOT="${RUN_ROOT:-runs/fremont_comparison}"
MASK_K="${MASK_K:-9}"
MASK_MODE="${MASK_MODE:-random}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
SEED="${SEED:-2026}"

COMMON_ARGS=(
  --data "$DATA_ROOT"
  --data-format incident
  --file-pattern 'incident_{split}.npy'
  --traffic-feature 0
  --graph-mode file
  --adjacency "$DATA_ROOT/adj_matrix.npy"
  --mask-mode "$MASK_MODE"
  --mask-k "$MASK_K"
  --dim 128
  --heads 4
  --layers 2
  --finetune-epochs "$FINETUNE_EPOCHS"
  --batch-size 32
  --lr 0.0001
  --seed "$SEED"
)

echo "[1/3] ST-GNN supervised"
python train.py "${COMMON_ARGS[@]}" \
  --pretrain-epochs 0 \
  --jepa-weight 0 \
  --finetune-strategy full \
  --output "$RUN_ROOT/stgnn_supervised"

echo "[2/3] JEPA + frozen probe"
python train.py "${COMMON_ARGS[@]}" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --jepa-weight 0 \
  --finetune-strategy head \
  --output "$RUN_ROOT/jepa_frozen_probe"

echo "[3/3] JEPA + full fine-tune"
python train.py "${COMMON_ARGS[@]}" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --jepa-weight 0.1 \
  --finetune-strategy full \
  --output "$RUN_ROOT/jepa_full_finetune"

python scripts/summarize_comparison.py "$RUN_ROOT"

