#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

FREMONT_ROOT="${FREMONT_ROOT:-/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Fremont}"
OAKLAND_ROOT="${OAKLAND_ROOT:-/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Oakland}"
RUN_ROOT="${RUN_ROOT:-runs/three_city_experiments}"
SEED="${SEED:-2026}"

FREMONT_ROOT="$FREMONT_ROOT" OUTPUT_DIR="$RUN_ROOT/exp1_fremont_fremont" \
SEED="$SEED" bash scripts/run_fremont_teacher_fremont_student.sh

OAKLAND_ROOT="$OAKLAND_ROOT" OUTPUT_DIR="$RUN_ROOT/oakland_teacher" \
SEED="$SEED" bash scripts/train_oakland_teacher.sh

TEACHER_CHECKPOINT="$RUN_ROOT/oakland_teacher/teacher/best_teacher.pt" \
FREMONT_ROOT="$FREMONT_ROOT" OUTPUT_DIR="$RUN_ROOT/exp2_oakland_fremont" \
SEED="$SEED" bash scripts/run_oakland_teacher_fremont_student.sh

TEACHER_CHECKPOINT="$RUN_ROOT/oakland_teacher/teacher/best_teacher.pt" \
SOURCE_ROOT="$OAKLAND_ROOT" TARGET_ROOT="$FREMONT_ROOT" \
OUTPUT_DIR="$RUN_ROOT/exp3_oakland_oakland_fremont" \
SEED="$SEED" bash scripts/run_oakland_to_fremont.sh

python summarize_experiments.py --root "$RUN_ROOT"
