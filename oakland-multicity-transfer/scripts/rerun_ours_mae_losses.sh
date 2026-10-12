#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_DIR/traffic_forcasting/TopoJEPA/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/oakland_multicity_seed2026_ours_rerun}"
DEVICE="${DEVICE:-cuda:1}"
SEED="${SEED:-2026}"

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --source-city Oakland
  --methods ours
  --seed "$SEED"
  --device "$DEVICE"
  --output "$OUTPUT_ROOT"
)

# These are exactly the city-horizon tasks where the first Ours run had a
# higher MAE than the corresponding STAEformer transfer result.
TASKS=(
  "6|Dublin,Livermore,San Leandro"
  "9|Berkeley,Newark,Pleasanton"
  "12|Dublin,Pleasanton,San Leandro"
)

cd "$PROJECT_DIR"

for task in "${TASKS[@]}"; do
  horizon="${task%%|*}"
  cities="${task#*|}"
  extra_args=()
  if [[ "${FORCE:-0}" == "1" ]]; then
    extra_args+=(--force)
  fi

  echo "===== rerun ours: horizon=$horizon targets=$cities seed=$SEED ====="
  python run.py \
    "${COMMON_ARGS[@]}" \
    --target-cities "$cities" \
    --horizons "$horizon" \
    "${extra_args[@]}"
done

python summarize.py --root "$OUTPUT_ROOT"

echo "Rerun results: $OUTPUT_ROOT/results.csv"
