#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_DIR/TopoJEPA/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/mask_ablation_three_counties_seed2026}"
DEVICE="${DEVICE:-cuda:1}"
SEED="${SEED:-2026}"
HORIZONS="${HORIZONS:-6,9,12}"
COUNTY_FILTER="${COUNTY_FILTER:-all}"

cd "$PROJECT_DIR"

ensure_splits() {
  local county_root="$1"
  local missing=0
  local existing=0
  for split in train val test; do
    if [[ -f "$county_root/incident_${split}.npy" ]]; then
      existing=$((existing + 1))
    else
      missing=$((missing + 1))
    fi
  done
  if [[ "$missing" -eq 3 ]]; then
    python "$REPO_DIR/Preprocess/split_county_incidents.py" \
      --county-root "$county_root"
  elif [[ "$missing" -ne 0 ]]; then
    echo "ERROR: partial train/val/test split in $county_root" >&2
    exit 2
  fi
}

prepare_county() {
  local county_root="$1"
  local cities="$2"
  ensure_splits "$county_root"
  python prepare_county_cities.py \
    --county-root "$county_root" \
    --output-root "$DATA_ROOT" \
    --cities "$cities" \
    --reuse-existing
}

run_county() {
  local county="$1"
  local source_city="$2"
  local targets="$3"
  python run_mask_ablation_multicounty.py \
    --data-root "$DATA_ROOT" \
    --county "$county" \
    --source-city "$source_city" \
    --target-cities "$targets" \
    --horizons "$HORIZONS" \
    --mask-blocks 8 \
    --seed "$SEED" \
    --device "$DEVICE" \
    --output "$OUTPUT_ROOT"
}

if [[ "$COUNTY_FILTER" == "all" || "$COUNTY_FILTER" == "alameda" ]]; then
  prepare_county "$DATA_ROOT/Alameda" "Oakland,Berkeley,Dublin"
  run_county "Alameda" "Oakland" "Berkeley,Dublin"
fi

if [[ "$COUNTY_FILTER" == "all" || "$COUNTY_FILTER" == "orange" ]]; then
  prepare_county "$DATA_ROOT/Orange" "Irvine,Laguna Niguel,Fullerton"
  run_county "Orange" "Irvine" "Laguna Niguel,Fullerton"
fi

if [[ "$COUNTY_FILTER" == "all" || "$COUNTY_FILTER" == "contra_costa" ]]; then
  prepare_county "$DATA_ROOT/Contra_Costa" "Richmond,Hercules,Pleasant Hill"
  run_county "Contra Costa" "Richmond" "Hercules,Pleasant Hill"
fi

python summarize_mask_ablation_multicounty.py --root "$OUTPUT_ROOT"

