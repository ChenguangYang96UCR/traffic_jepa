#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_DIR/traffic_forcasting/TopoJEPA/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/oakland_multicity_seed2026}"
DEVICE="${DEVICE:-cuda:1}"
SEED="${SEED:-2026}"
MASK_MODE="topology_st"

cd "$PROJECT_DIR"
bash scripts/fetch_ts_jepa.sh
python run.py \
  --data-root "$DATA_ROOT" \
  --source-city Oakland \
  --target-cities "Berkeley,Dublin,Newark,San Leandro,Pleasanton,Livermore" \
  --methods "ours,pdformer,flashst,testam,patchstg,staeformer,scjepa,ts_jepa,staeformer_jepa,dcrnn,agcrn,sdgformer,dstfgcn" \
  --horizons "6,9,12" \
  --mask-mode "$MASK_MODE" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --output "$OUTPUT_ROOT"

python summarize.py --root "$OUTPUT_ROOT"
