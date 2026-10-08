#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

: "${SOURCE_ROOT:?Set SOURCE_ROOT to the Oakland dataset directory}"
SOURCE_ADJ="${SOURCE_ADJ:-$SOURCE_ROOT/adj_matrix.npy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/shared_teacher_backbones}"
SHARED_TEACHER_ROOT="${SHARED_TEACHER_ROOT:-$OUTPUT_ROOT/shared_teacher}"
SEED="${SEED:-2026}"
HORIZON="${HORIZON:-12}"
TEACHER_OUTPUT="$SHARED_TEACHER_ROOT/horizon_$HORIZON/seed_$SEED"

python train_shared_teacher.py \
  --data "$SOURCE_ROOT" \
  --adj "$SOURCE_ADJ" \
  --output "$TEACHER_OUTPUT" \
  --input-steps 12 \
  --pred-steps "$HORIZON" \
  --latent-dim "${TEACHER_DIM:-128}" \
  --layers "${TEACHER_LAYERS:-3}" \
  --heads "${TEACHER_HEADS:-8}" \
  --ff-dim "${TEACHER_FF_DIM:-512}" \
  --graph-pe-dim "${GRAPH_PE_DIM:-16}" \
  --epochs "${TEACHER_EPOCHS:-50}" \
  --lr "${TEACHER_LR:-1e-3}" \
  --mask-blocks "${MASK_BLOCKS:-8}" \
  --future-block-ratio "${TEACHER_FUTURE_BLOCK_RATIO:-0.5}" \
  --mask-min-time "${MASK_MIN_TIME:-2}" \
  --mask-max-time "${MASK_MAX_TIME:-6}" \
  --mask-min-sensor-ratio "${MASK_MIN_SENSOR_RATIO:-0.10}" \
  --mask-max-sensor-ratio "${MASK_MAX_SENSOR_RATIO:-0.30}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --patience "${PATIENCE:-10}" \
  --workers "${WORKERS:-0}" \
  --seed "$SEED" \
  --device "${DEVICE:-cuda:2}"
