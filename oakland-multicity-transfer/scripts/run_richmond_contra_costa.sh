#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_DIR/traffic_forcasting/TopoJEPA/dataset}"
CONTRA_ROOT="${CONTRA_ROOT:-$DATA_ROOT/Contra_Costa}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/richmond_contra_costa_multicity_seed2026}"
DEVICE="${DEVICE:-cuda:1}"
SEED="${SEED:-2026}"
MASK_MODE="topology_st"
METHODS="${METHODS:-ours,pdformer,flashst,testam,patchstg,scjepa,ts_jepa,dcrnn,agcrn,sdgformer,dstfgcn}"
CITIES="Richmond,Hercules,Pleasant Hill,Orinda,Lafayette,San Ramon,Martinez"
TARGETS="Hercules,Pleasant Hill,Orinda,Lafayette,San Ramon,Martinez"

cd "$PROJECT_DIR"

# The released county container contains incident_all.npy.  Create one shared,
# chronological 70/15/15 split before deriving node-aligned city views.
missing_splits=0
existing_splits=0
for split in train val test; do
  if [[ -f "$CONTRA_ROOT/incident_${split}.npy" ]]; then
    existing_splits=$((existing_splits + 1))
  else
    missing_splits=$((missing_splits + 1))
  fi
done

if [[ "$missing_splits" -eq 3 ]]; then
  python "$REPO_DIR/Preprocess/split_county_incidents.py" \
    --county-root "$CONTRA_ROOT"
elif [[ "$missing_splits" -ne 0 ]]; then
  echo "ERROR: Contra_Costa has a partial train/val/test split." >&2
  echo "Remove or verify the partial files, then rerun with an explicit --overwrite split." >&2
  exit 2
fi

python prepare_county_cities.py \
  --county-root "$CONTRA_ROOT" \
  --output-root "$DATA_ROOT" \
  --cities "$CITIES" \
  --reuse-existing

bash scripts/fetch_ts_jepa.sh

python run.py \
  --data-root "$DATA_ROOT" \
  --source-city Richmond \
  --target-cities "$TARGETS" \
  --methods "$METHODS" \
  --horizons "6,9,12" \
  --mask-mode "$MASK_MODE" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --output "$OUTPUT_ROOT"

python summarize.py --root "$OUTPUT_ROOT"
