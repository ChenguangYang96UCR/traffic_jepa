#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
OAKLAND_ROOT="${OAKLAND_ROOT:-../TopoJEPA/dataset/Oakland}"
OAKLAND_TEACHER_CHECKPOINT="${OAKLAND_TEACHER_CHECKPOINT:-runs/oakland_teacher/teacher/best_teacher.pt}"
RUN_ROOT="${RUN_ROOT:-runs/three_berkeley_experiments}"
SEED="${SEED:-2026}"

if [[ ! -f "$OAKLAND_TEACHER_CHECKPOINT" ]]; then
  echo "ERROR: pre-trained Oakland Teacher not found: $OAKLAND_TEACHER_CHECKPOINT" >&2
  echo "Set OAKLAND_TEACHER_CHECKPOINT to the existing best_teacher.pt." >&2
  exit 2
fi

# 1. Berkeley Teacher -> Berkeley Student -> Berkeley downstream.
BERKELEY_ROOT="$BERKELEY_ROOT" \
OUTPUT_DIR="$RUN_ROOT/exp1_berkeley_berkeley" \
SEED="$SEED" \
bash scripts/run_berkeley_teacher_berkeley_student.sh

# 2. Shared frozen Oakland Teacher -> Berkeley Student -> Berkeley downstream.
BERKELEY_ROOT="$BERKELEY_ROOT" \
TEACHER_CHECKPOINT="$OAKLAND_TEACHER_CHECKPOINT" \
OUTPUT_DIR="$RUN_ROOT/exp2_oakland_berkeley" \
SEED="$SEED" \
bash scripts/run_oakland_teacher_berkeley_student.sh

# 3. Shared frozen Oakland Teacher -> Oakland Student -> Berkeley transfer.
OAKLAND_ROOT="$OAKLAND_ROOT" \
BERKELEY_ROOT="$BERKELEY_ROOT" \
TEACHER_CHECKPOINT="$OAKLAND_TEACHER_CHECKPOINT" \
OUTPUT_DIR="$RUN_ROOT/exp3_oakland_oakland_berkeley" \
SEED="$SEED" \
bash scripts/run_oakland_to_berkeley.sh

python summarize_experiments.py \
  --root "$RUN_ROOT" \
  --target-city Berkeley \
  --source-city Oakland
