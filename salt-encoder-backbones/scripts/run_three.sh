#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the frozen Oakland d=128 Teacher}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the Oakland dataset directory}"
: "${TARGET_ROOT:?Set TARGET_ROOT to the Berkeley dataset directory}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/oakland_teacher_berkeley_backbones}"
export TEACHER_CHECKPOINT SOURCE_ROOT TARGET_ROOT OUTPUT_ROOT

for model in patchtst stgformer stlaformer; do
  echo "===== $model ====="
  MODEL="$model" bash scripts/run_one.sh
done

python summarize.py --root "$OUTPUT_ROOT" | tee "$OUTPUT_ROOT/summary.txt"
