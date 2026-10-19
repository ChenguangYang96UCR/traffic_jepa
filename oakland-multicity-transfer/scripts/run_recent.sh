#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_DIR/TopoJEPA/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/oakland_multicity_seed2026}"
DEVICE="${DEVICE:-cuda:1}"

cd "$PROJECT_DIR"
python run.py \
  --data-root "$DATA_ROOT" \
  --source-city Oakland \
  --target-cities "Berkeley,Dublin,Newark,San Leandro,Pleasanton,Livermore" \
  --methods "sdgformer,dstfgcn" \
  --horizons "6,9,12" \
  --seed 2026 \
  --device "$DEVICE" \
  --output "$OUTPUT_ROOT"

python summarize.py --root "$OUTPUT_ROOT"
