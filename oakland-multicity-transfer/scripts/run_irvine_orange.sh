#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_DIR/TopoJEPA/dataset}"
ORANGE_ROOT="${ORANGE_ROOT:-$DATA_ROOT/Orange}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/irvine_orange_multicity_seed2026}"
DEVICE="${DEVICE:-cuda:1}"
SEED="${SEED:-2026}"
MASK_MODE="topology_st"
METHODS="${METHODS:-ours,pdformer,flashst,testam,patchstg,scjepa,ts_jepa,dcrnn,agcrn,sdgformer,dstfgcn}"
CITIES="Irvine,Laguna Niguel,Newport Beach,Westminster,Fullerton,Buena Park,Mission Viejo"
TARGETS="Laguna Niguel,Newport Beach,Westminster,Fullerton,Buena Park,Mission Viejo"

cd "$PROJECT_DIR"
python prepare_county_cities.py \
  --county-root "$ORANGE_ROOT" \
  --output-root "$DATA_ROOT" \
  --cities "$CITIES" \
  --reuse-existing

bash scripts/fetch_ts_jepa.sh
python run.py \
  --data-root "$DATA_ROOT" \
  --source-city Irvine \
  --target-cities "$TARGETS" \
  --methods "$METHODS" \
  --horizons "6,9,12" \
  --mask-mode "$MASK_MODE" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --output "$OUTPUT_ROOT"

python summarize.py --root "$OUTPUT_ROOT"
