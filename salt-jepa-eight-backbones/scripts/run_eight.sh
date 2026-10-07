#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# MODELS=(pdformer flashst patchstg testam patchtst staeformer stgormer tsformer)
MODELS=(stgormer tsformer)
read -r -a SEED_LIST <<< "${SEEDS:-2024 8888 1}"
if [[ "${#SEED_LIST[@]}" -ne 3 && -z "${ALLOW_NONTHREE_SEEDS:-}" ]]; then
  echo "ERROR: exactly three seeds are required; got: ${SEED_LIST[*]}" >&2
  echo "Set ALLOW_NONTHREE_SEEDS=1 only for smoke tests." >&2
  exit 2
fi

for seed in "${SEED_LIST[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "===== seed=$seed model=$model ====="
    SEED="$seed" MODEL="$model" bash scripts/run_one.sh
  done
done

python summarize.py \
  --root "${OUTPUT_ROOT:-runs/eight_backbones}" \
  --expected-seeds "${#SEED_LIST[@]}"
