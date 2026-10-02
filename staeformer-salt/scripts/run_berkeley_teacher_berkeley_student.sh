#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/exp1_berkeley_berkeley}"
SEED="${SEED:-2026}"

for split in train val test; do
  file="$BERKELEY_ROOT/incident_${split}.npy"
  if [[ ! -f "$file" ]]; then
    echo "ERROR: Berkeley split not found: $file" >&2
    exit 2
  fi
done

python run.py \
  --pipeline in_domain \
  --mode all \
  --experiment-label "Berkeley Teacher -> Berkeley Student" \
  --teacher-data "$BERKELEY_ROOT" \
  --student-data "$BERKELEY_ROOT" \
  --target-data "$BERKELEY_ROOT" \
  --teacher-sensor-dim 0 \
  --input-steps 12 \
  --pred-steps 12 \
  --teacher-epochs 20 \
  --student-epochs 50 \
  --finetune-epochs 50 \
  --teacher-lr 0.001 \
  --student-lr 0.0001 \
  --finetune-lr 0.001 \
  --encoder-lr-scale 0.1 \
  --batch-size 16 \
  --patience 10 \
  --seed "$SEED" \
  --output "$OUTPUT_DIR"
