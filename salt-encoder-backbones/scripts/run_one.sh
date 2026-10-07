#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

: "${MODEL:?Set MODEL to patchtst, stgformer, or stlaformer}"
: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the frozen Oakland d=128 Teacher}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the Oakland dataset directory}"
: "${TARGET_ROOT:?Set TARGET_ROOT to the Berkeley dataset directory}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/oakland_teacher_berkeley_backbones}"
SEED="${SEED:-2026}"

python run.py \
  --model "$MODEL" \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --source-data "$SOURCE_ROOT" \
  --target-data "$TARGET_ROOT" \
  --output "$OUTPUT_ROOT/$MODEL" \
  --input-steps 12 \
  --pred-steps 12 \
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
  --device "${DEVICE:-cuda}" \
  ${SKIP:+--skip "$SKIP"}
