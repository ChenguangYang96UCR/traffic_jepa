#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_DIR/traffic_forcasting/TopoJEPA/dataset}"
ORANGE_ROOT="${ORANGE_ROOT:-$DATA_ROOT/Orange}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/irvine_mask_ablation_seed2026}"
RANDOM_SOURCE_CACHE_ROOT="${RANDOM_SOURCE_CACHE_ROOT:-$PROJECT_DIR/runs/irvine_orange_multicity_seed2026}"
DEVICE="${DEVICE:-cuda:1}"
SEED="${SEED:-2026}"
TARGET_TRAIN_FRACTION="${TARGET_TRAIN_FRACTION:-1.0}"
CITIES="Irvine,Laguna Niguel,Newport Beach,Westminster,Fullerton,Buena Park,Mission Viejo"
TARGETS="${TARGETS:-Laguna Niguel,Newport Beach,Westminster,Fullerton,Buena Park,Mission Viejo}"

cd "$PROJECT_DIR"

python prepare_county_cities.py \
  --county-root "$ORANGE_ROOT" \
  --output-root "$DATA_ROOT" \
  --cities "$CITIES" \
  --reuse-existing

for mask_mode in random_st topology_st; do
  run_output="$OUTPUT_ROOT/$mask_mode"
  if [[ "$mask_mode" == "random_st" ]]; then
    source_cache="$RANDOM_SOURCE_CACHE_ROOT"
  else
    source_cache="$run_output"
  fi

  python run.py \
    --data-root "$DATA_ROOT" \
    --source-city Irvine \
    --target-cities "$TARGETS" \
    --methods ours \
    --horizons "6,9,12" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --source-cache-root "$source_cache" \
    --target-train-fraction "$TARGET_TRAIN_FRACTION" \
    --mask-mode "$mask_mode" \
    --mask-blocks 8 \
    --future-block-ratio 0.5 \
    --output "$run_output"
done

python summarize_mask_ablation.py --root "$OUTPUT_ROOT"
