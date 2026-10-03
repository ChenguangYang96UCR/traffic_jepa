#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OAKLAND_ROOT="${OAKLAND_ROOT:-../TopoJEPA/dataset/Oakland}"
BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/oakland_supervised_to_berkeley}"
SALT_ROOT="${SALT_ROOT:-runs/future_focus_berkeley}"
SEED="${SEED:-2026}"

for root in "$OAKLAND_ROOT" "$BERKELEY_ROOT"; do
  for split in train val test; do
    file="$root/incident_${split}.npy"
    if [[ ! -f "$file" ]]; then
      echo "ERROR: dataset split not found: $file" >&2
      exit 2
    fi
  done
done

python run_supervised_transfer.py \
  --source-data "$OAKLAND_ROOT" \
  --target-data "$BERKELEY_ROOT" \
  --input-steps 12 \
  --pred-steps 12 \
  --source-epochs 50 \
  --finetune-epochs 50 \
  --source-lr 0.001 \
  --finetune-lr 0.001 \
  --encoder-lr-scale 0.1 \
  --batch-size 16 \
  --patience 10 \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"

SALT_ALL="$SALT_ROOT/exp3_oakland_oakland_berkeley/all/all_results.json"
SALT_FUTURE="$SALT_ROOT/exp3_oakland_oakland_berkeley/future/all_results.json"
if [[ -f "$SALT_ALL" && -f "$SALT_FUTURE" ]]; then
  python compare_supervised_salt.py \
    --supervised-root "$OUTPUT_DIR" \
    --salt-root "$SALT_ROOT"
else
  echo "SALT result files were not found under $SALT_ROOT; skipping merged comparison."
  echo "The supervised-transfer result is still available at $OUTPUT_DIR/summary.txt."
fi
