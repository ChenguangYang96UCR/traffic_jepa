#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

: "${MODEL:?Set MODEL to pdformer, flashst, patchstg, testam, patchtst, staeformer, stgormer, or tsformer}"
: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the frozen Oakland d=128 Teacher}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the Oakland dataset directory}"
: "${TARGET_ROOT:?Set TARGET_ROOT to the Berkeley dataset directory}"

SOURCE_ADJ="${SOURCE_ADJ:-$SOURCE_ROOT/adj_matrix.npy}"
TARGET_ADJ="${TARGET_ADJ:-$TARGET_ROOT/adj_matrix.npy}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/eight_backbones}"
SEED="${SEED:-2026}"
HORIZON="${HORIZON:-12}"
if [[ -n "${HORIZON_SWEEP:-}" ]]; then
  RUN_OUTPUT="$OUTPUT_ROOT/$MODEL/horizon_$HORIZON/seed_$SEED"
else
  RUN_OUTPUT="$OUTPUT_ROOT/$MODEL/seed_$SEED"
fi
EXTRA_ARGS=()
if [[ -n "${SKIP:-}" ]]; then
  EXTRA_ARGS+=(--skip "$SKIP")
fi

python run.py \
  --model "$MODEL" \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --expected-teacher-mask-blocks "${EXPECTED_TEACHER_MASK_BLOCKS:-0}" \
  --expected-teacher-future-block-ratio "${EXPECTED_TEACHER_FUTURE_BLOCK_RATIO:--1}" \
  --source-data "$SOURCE_ROOT" \
  --target-data "$TARGET_ROOT" \
  --source-adj "$SOURCE_ADJ" \
  --target-adj "$TARGET_ADJ" \
  --output "$RUN_OUTPUT" \
  --input-steps 12 \
  --pred-steps "$HORIZON" \
  --student-epochs "${STUDENT_EPOCHS:-100}" \
  --forecast-epochs "${FORECAST_EPOCHS:-100}" \
  --source-epochs "${SOURCE_EPOCHS:-100}" \
  --student-lr "${STUDENT_LR:-1e-4}" \
  --finetune-lr "${FINETUNE_LR:-1e-3}" \
  --encoder-lr-scale "${ENCODER_LR_SCALE:-0.1}" \
  --distill-loss "${DISTILL_LOSS:-l2}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --patience "${PATIENCE:-10}" \
  --workers "${WORKERS:-0}" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda:1}" \
  "${EXTRA_ARGS[@]}"

python run_jepa.py \
  --model "$MODEL" \
  --target-data "$TARGET_ROOT" \
  --target-adj "$TARGET_ADJ" \
  --output "$RUN_OUTPUT" \
  --input-steps 12 \
  --pred-steps "$HORIZON" \
  --pretrain-epochs "${JEPA_EPOCHS:-100}" \
  --forecast-epochs "${FORECAST_EPOCHS:-100}" \
  --pretrain-lr "${JEPA_LR:-1e-4}" \
  --finetune-lr "${FINETUNE_LR:-1e-3}" \
  --encoder-lr-scale "${ENCODER_LR_SCALE:-0.1}" \
  --latent-loss "${JEPA_LOSS:-l2}" \
  --ema-momentum "${EMA_MOMENTUM:-0.996}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --patience "${PATIENCE:-10}" \
  --workers "${WORKERS:-0}" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda:1}"
