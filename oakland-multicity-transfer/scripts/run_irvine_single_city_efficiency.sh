#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_DIR/traffic_forcasting/TopoJEPA/dataset}"
ORANGE_ROOT="${ORANGE_ROOT:-$DATA_ROOT/Orange}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/irvine_laguna_niguel_h12_efficiency_seed2026}"
# Reuse already-trained Irvine source representations when available.
SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT:-$PROJECT_DIR/runs/irvine_orange_multicity_seed2026}"
DEVICE="${DEVICE:-cuda:1}"
SEED="${SEED:-2026}"
MASK_MODE="topology_st"
HORIZON="${HORIZON:-12}"
TARGET_CITY="${TARGET_CITY:-Laguna Niguel}"
METHODS="${METHODS:-ours,pdformer,flashst,testam,patchstg,staeformer,scjepa,ts_jepa,staeformer_jepa,dcrnn,agcrn,sdgformer,dstfgcn}"
WARMUP="${WARMUP:-50}"
REPEATS="${REPEATS:-200}"
BATCH_SIZES="${BATCH_SIZES:-1,32}"
PROFILE_ONLY="${PROFILE_ONLY:-0}"

cd "$PROJECT_DIR"

if [[ "$PROFILE_ONLY" != "1" && "$METHODS" == *"ts_jepa"* ]]; then
  bash scripts/fetch_ts_jepa.sh
fi

if [[ "$PROFILE_ONLY" != "1" ]]; then
  python run.py \
    --data-root "$DATA_ROOT" \
    --source-city Irvine \
    --target-cities "$TARGET_CITY" \
    --methods "$METHODS" \
    --horizons "$HORIZON" \
    --mask-mode "$MASK_MODE" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --source-cache-root "$SOURCE_CACHE_ROOT" \
    --output "$OUTPUT_ROOT"
fi

python profile_efficiency.py \
  --data-root "$DATA_ROOT" \
  --run-root "$OUTPUT_ROOT" \
  --target-city "$TARGET_CITY" \
  --methods "$METHODS" \
  --horizon "$HORIZON" \
  --device "$DEVICE" \
  --batch-sizes "$BATCH_SIZES" \
  --warmup "$WARMUP" \
  --repeats "$REPEATS"
