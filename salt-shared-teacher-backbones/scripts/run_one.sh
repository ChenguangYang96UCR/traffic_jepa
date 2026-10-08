#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

: "${MODEL:?Set MODEL to pdformer, flashst, patchstg, testam, patchtst, staeformer, stgormer, or tsformer}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the Oakland dataset directory}"
: "${TARGET_ROOT:?Set TARGET_ROOT to the Berkeley dataset directory}"

SOURCE_ADJ="${SOURCE_ADJ:-$SOURCE_ROOT/adj_matrix.npy}"
TARGET_ADJ="${TARGET_ADJ:-$TARGET_ROOT/adj_matrix.npy}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/shared_teacher_backbones}"
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
if [[ ",${SKIP:-}," != *,salt,* ]]; then
  if [[ -n "${TEACHER_CHECKPOINT:-}" ]]; then
    SHARED_TEACHER="$TEACHER_CHECKPOINT"
  else
    SHARED_TEACHER_ROOT="${SHARED_TEACHER_ROOT:-$OUTPUT_ROOT/shared_teacher}"
    SHARED_TEACHER="$SHARED_TEACHER_ROOT/horizon_$HORIZON/seed_$SEED/best_teacher.pt"
  fi
  if [[ ! -f "$SHARED_TEACHER" ]]; then
    echo "ERROR: shared Teacher checkpoint not found: $SHARED_TEACHER" >&2
    echo "Run scripts/train_shared_teacher.sh first." >&2
    exit 2
  fi
  EXTRA_ARGS+=(--teacher-checkpoint "$SHARED_TEACHER")
fi

python run.py \
  --model "$MODEL" \
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
  --distill-loss "${DISTILL_LOSS:-l1}" \
  --mask-blocks "${MASK_BLOCKS:-8}" \
  --teacher-future-block-ratio "${TEACHER_FUTURE_BLOCK_RATIO:-0.5}" \
  --mask-min-time "${MASK_MIN_TIME:-2}" \
  --mask-max-time "${MASK_MAX_TIME:-6}" \
  --mask-min-sensor-ratio "${MASK_MIN_SENSOR_RATIO:-0.10}" \
  --mask-max-sensor-ratio "${MASK_MAX_SENSOR_RATIO:-0.30}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --patience "${PATIENCE:-10}" \
  --workers "${WORKERS:-0}" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda:2}" \
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
  --device "${DEVICE:-cuda:2}"
