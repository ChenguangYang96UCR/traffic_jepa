#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 METHOD TARGET_CITY HORIZON" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/TopoJEPA/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/oakland_multicity_seed2026}"
DEVICE="${DEVICE:-cuda:1}"

cd "$PROJECT_DIR"
if [[ "$1" == "ts_jepa" ]]; then
  bash scripts/fetch_ts_jepa.sh
fi
python run.py --data-root "$DATA_ROOT" --source-city Oakland \
  --target-cities "$2" --methods "$1" --horizons "$3" \
  --seed 2026 --device "$DEVICE" --output "$OUTPUT_ROOT"
python summarize.py --root "$OUTPUT_ROOT"
