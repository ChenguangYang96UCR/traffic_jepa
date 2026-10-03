#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
OAKLAND_ROOT="${OAKLAND_ROOT:-../TopoJEPA/dataset/Oakland}"
RUN_ROOT="${RUN_ROOT:-runs/future_focus_berkeley}"
SEED="${SEED:-2026}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-50}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
FUTURE_BLOCK_RATIO="${FUTURE_BLOCK_RATIO:-0.75}"
MASK_BLOCKS="${MASK_BLOCKS:-4}"

BERKELEY_TEACHER_CHECKPOINT="${BERKELEY_TEACHER_CHECKPOINT:-$RUN_ROOT/teachers/berkeley/teacher/best_teacher.pt}"
OAKLAND_TEACHER_CHECKPOINT="${OAKLAND_TEACHER_CHECKPOINT:-$RUN_ROOT/teachers/oakland/teacher/best_teacher.pt}"

validate_dataset() {
  local root="$1" city="$2"
  for split in train val test; do
    local file="$root/incident_${split}.npy"
    if [[ ! -f "$file" ]]; then
      echo "ERROR: $city split not found: $file" >&2
      exit 2
    fi
  done
}

train_future_teacher() {
  local city="$1" root="$2" output="$3" checkpoint="$4"
  if [[ -f "$checkpoint" ]]; then
    echo "Reuse future-focused $city Teacher: $checkpoint"
    return
  fi
  python run.py \
    --mode teacher \
    --teacher-data "$root" \
    --teacher-sensor-dim 0 \
    --teacher-mask-policy future_biased \
    --teacher-future-block-ratio "$FUTURE_BLOCK_RATIO" \
    --mask-blocks "$MASK_BLOCKS" \
    --input-steps 12 \
    --pred-steps 12 \
    --teacher-epochs "$TEACHER_EPOCHS" \
    --teacher-lr 0.001 \
    --batch-size 16 \
    --patience 10 \
    --seed "$SEED" \
    --output "$output"
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: expected Teacher checkpoint was not produced: $checkpoint" >&2
    exit 2
  fi
}

run_student_scope() {
  local pipeline="$1" teacher="$2" student_root="$3" target_root="$4"
  local output="$5" scope="$6" strategies="$7" label="$8"
  python run.py \
    --pipeline "$pipeline" \
    --mode student_downstream \
    --experiment-label "$label | Student loss=$scope" \
    --teacher-checkpoint "$teacher" \
    --require-future-focused-teacher \
    --student-data "$student_root" \
    --target-data "$target_root" \
    --student-distill-scope "$scope" \
    --downstream-strategies "$strategies" \
    --input-steps 12 \
    --pred-steps 12 \
    --student-epochs "$STUDENT_EPOCHS" \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --student-lr 0.0001 \
    --finetune-lr 0.001 \
    --encoder-lr-scale 0.1 \
    --batch-size 16 \
    --patience 10 \
    --seed "$SEED" \
    --output "$output"
}

validate_dataset "$BERKELEY_ROOT" Berkeley
validate_dataset "$OAKLAND_ROOT" Oakland

# The Teacher objective changed, so legacy uniform-mask checkpoints are not a
# valid controlled comparison. Train each future-focused Teacher once and share
# it across both Student loss scopes.
train_future_teacher Berkeley "$BERKELEY_ROOT" \
  "$RUN_ROOT/teachers/berkeley" "$BERKELEY_TEACHER_CHECKPOINT"
train_future_teacher Oakland "$OAKLAND_ROOT" \
  "$RUN_ROOT/teachers/oakland" "$OAKLAND_TEACHER_CHECKPOINT"

# Train the common scratch baseline only once. Every downstream target is the
# same Berkeley split, so repeating scratch would only waste compute.
run_student_scope in_domain "$BERKELEY_TEACHER_CHECKPOINT" \
  "$BERKELEY_ROOT" "$BERKELEY_ROOT" \
  "$RUN_ROOT/exp1_berkeley_berkeley/all" all "scratch,frozen,full" \
  "Berkeley Teacher -> Berkeley Student"
run_student_scope in_domain "$BERKELEY_TEACHER_CHECKPOINT" \
  "$BERKELEY_ROOT" "$BERKELEY_ROOT" \
  "$RUN_ROOT/exp1_berkeley_berkeley/future" future "frozen,full" \
  "Berkeley Teacher -> Berkeley Student"

for scope in all future; do
  run_student_scope in_domain "$OAKLAND_TEACHER_CHECKPOINT" \
    "$BERKELEY_ROOT" "$BERKELEY_ROOT" \
    "$RUN_ROOT/exp2_oakland_berkeley/$scope" "$scope" "frozen,full" \
    "Oakland Teacher -> Berkeley Student"

  run_student_scope cross_city "$OAKLAND_TEACHER_CHECKPOINT" \
    "$OAKLAND_ROOT" "$BERKELEY_ROOT" \
    "$RUN_ROOT/exp3_oakland_oakland_berkeley/$scope" "$scope" "frozen,full" \
    "Oakland Teacher -> Oakland Student -> Berkeley"
done

python summarize_future_focus.py --root "$RUN_ROOT"
