#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_DIR/traffic_forcasting/TopoJEPA/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/salt_Alameda_backbone_teacher_seed2026}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
HORIZONS="${HORIZONS:-6,9,12}"
# BACKBONES="${BACKBONES:-pdformer,flashst,testam,patchstg,dcrnn,agcrn,sdgformer,dstfgcn}"
BACKBONES="${BACKBONES:-pdformer,flashst,testam}"
MASK_MODE="topology_st"
COUNTY_FILTER="${COUNTY_FILTER:-all}"
# Colon-separated existing result files. They are read only by the final
# summarizer; no baseline is retrained by this script.
BASELINE_CSVS="${BASELINE_CSVS:-$PROJECT_DIR/results.csv}"

cd "$PROJECT_DIR"

run_county() {
  local county="$1"
  local source_city="$2"
  local targets="$3"
  python run_salt_multicounty.py \
    --data-root "$DATA_ROOT" \
    --county "$county" \
    --source-city "$source_city" \
    --target-cities "$targets" \
    --backbones "$BACKBONES" \
    --horizons "$HORIZONS" \
    --mask-mode "$MASK_MODE" \
    --mask-blocks 8 \
    --future-block-ratio 0.5 \
    --seed "$SEED" \
    --device "$DEVICE" \
    --output "$OUTPUT_ROOT"
}


if [[ "$COUNTY_FILTER" == "all" || "$COUNTY_FILTER" == "alameda" ]]; then
  run_county "Alameda" "Oakland" "Berkeley,Dublin"
fi

summary_args=(--root "$OUTPUT_ROOT")
IFS=':' read -r -a baseline_files <<< "$BASELINE_CSVS"
for baseline_file in "${baseline_files[@]}"; do
  if [[ -n "$baseline_file" && -f "$baseline_file" ]]; then
    summary_args+=(--baseline-csv "$baseline_file")
  fi
done
python summarize_salt_multicounty.py "${summary_args[@]}"
