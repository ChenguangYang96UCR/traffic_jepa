#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Same complete SC-JEPA method, architecture, loss weights and 100-epoch
# protocol as the paper-method run. Only the optimizer's numerical settings
# are adapted for the much longer Fremont update stream.
bash run_fremont_paper_method.sh \
  --lr 0.0001 \
  --optimizer-eps 0.000001 \
  --run-dir runs/fremont_paper_method_stable_seed2026 \
  "$@"
