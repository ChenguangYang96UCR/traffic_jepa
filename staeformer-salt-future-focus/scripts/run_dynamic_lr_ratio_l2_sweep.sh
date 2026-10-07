#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OAKLAND_ROOT="${OAKLAND_ROOT:-../TopoJEPA/dataset/Oakland}"
BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
RUN_ROOT="${RUN_ROOT:-runs/dynamic_lr_ratio_l2_sweep}"
BASE_SWEEP_ROOT="${BASE_SWEEP_ROOT:-runs/oakland_teacher128_berkeley_sweep}"
RATIOS="${RATIOS:-0.25 0.375 0.5 0.625 0.75}"
SCHEDULERS="${SCHEDULERS:-constant cosine onecycle plateau}"
DISTILL_LOSSES="${DISTILL_LOSSES:-l1 l2}"
SEED="${SEED:-2026}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-50}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATIENCE="${PATIENCE:-10}"
FINETUNE_LR="${FINETUNE_LR:-0.001}"
FINETUNE_MIN_LR="${FINETUNE_MIN_LR:-0.000001}"

for root in "$OAKLAND_ROOT" "$BERKELEY_ROOT"; do
  for split in train val test; do
    test -f "$root/incident_${split}.npy" || {
      echo "ERROR: missing $root/incident_${split}.npy" >&2
      exit 2
    }
  done
done

mkdir -p "$RUN_ROOT"

teacher_checkpoint() {
  local ratio="$1"
  local legacy="$BASE_SWEEP_ROOT/teachers/blocks=8_ratio=$ratio/teacher/best_teacher.pt"
  local output="$RUN_ROOT/teachers/ratio=$ratio"
  local checkpoint="$output/teacher/best_teacher.pt"
  if [[ -f "$legacy" ]]; then
    echo "$legacy"
    return
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "=== Train Oakland Teacher: blocks=8 ratio=$ratio ===" >&2
    python run.py \
      --mode teacher \
      --teacher-data "$OAKLAND_ROOT" \
      --teacher-input-dim 64 --teacher-step-dim 64 --teacher-sensor-dim 0 \
      --teacher-ff-dim 256 --teacher-heads 4 --teacher-layers 3 \
      --teacher-mask-policy future_biased \
      --teacher-future-block-ratio "$ratio" \
      --mask-blocks 8 --input-steps 12 --pred-steps 12 \
      --teacher-epochs "$TEACHER_EPOCHS" --teacher-lr 0.001 \
      --batch-size "$BATCH_SIZE" --patience "$PATIENCE" \
      --seed "$SEED" --output "$output" >&2
  fi
  echo "$checkpoint"
}

student_checkpoint() {
  local ratio="$1"
  local loss="$2"
  local teacher="$3"
  local legacy="$BASE_SWEEP_ROOT/students/blocks=8_ratio=$ratio/student/best_student.pt"
  local output="$RUN_ROOT/students/ratio=$ratio/loss=$loss"
  local checkpoint="$output/student/best_student.pt"
  if [[ "$loss" == "l1" && -f "$legacy" ]]; then
    echo "$legacy"
    return
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "=== Train Berkeley Student: ratio=$ratio latent=$loss ===" >&2
    python run.py \
      --mode student \
      --teacher-checkpoint "$teacher" \
      --expected-teacher-dim 128 \
      --expected-teacher-mask-policy future_biased \
      --expected-teacher-future-block-ratio "$ratio" \
      --expected-teacher-mask-blocks 8 \
      --student-data "$BERKELEY_ROOT" \
      --student-distill-scope all \
      --student-distill-loss "$loss" \
      --input-steps 12 --pred-steps 12 \
      --student-epochs "$STUDENT_EPOCHS" --student-lr 0.0001 \
      --batch-size "$BATCH_SIZE" --patience "$PATIENCE" \
      --seed "$SEED" --output "$output" >&2
  fi
  echo "$checkpoint"
}

run_result() {
  local ratio="$1"
  local loss="$2"
  local scheduler="$3"
  local student="$4"
  local output="$RUN_ROOT/results/ratio=$ratio/loss=$loss/scheduler=$scheduler"
  if [[ -f "$output/all_results.json" ]]; then
    echo "Reusing $output"
    return
  fi
  echo "=== Full FT: ratio=$ratio latent=$loss scheduler=$scheduler ==="
  python run.py \
    --pipeline in_domain \
    --mode downstream \
    --experiment-label "Oakland T128 -> Berkeley S | ratio=$ratio latent=$loss scheduler=$scheduler" \
    --student-checkpoint "$student" \
    --student-data "$BERKELEY_ROOT" --target-data "$BERKELEY_ROOT" \
    --student-distill-scope all --student-distill-loss "$loss" \
    --downstream-strategies full \
    --input-steps 12 --pred-steps 12 \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --finetune-lr "$FINETUNE_LR" \
    --finetune-min-lr "$FINETUNE_MIN_LR" \
    --finetune-lr-scheduler "$scheduler" \
    --encoder-lr-scale 0.1 \
    --forecast-loss mae \
    --batch-size "$BATCH_SIZE" --patience "$PATIENCE" \
    --seed "$SEED" --output "$output"
}

# A. Dynamic learning-rate policies at the reported best mask configuration.
teacher="$(teacher_checkpoint 0.5)"
student="$(student_checkpoint 0.5 l1 "$teacher")"
for scheduler in $SCHEDULERS; do
  run_result 0.5 l1 "$scheduler" "$student"
done

# B. Future-block allocation. Constant LR isolates the ratio effect.
for ratio in $RATIOS; do
  teacher="$(teacher_checkpoint "$ratio")"
  student="$(student_checkpoint "$ratio" l1 "$teacher")"
  run_result "$ratio" l1 constant "$student"
done

# C. Latent L1 vs latent L2. Forecast fine-tuning still uses MAE.
teacher="$(teacher_checkpoint 0.5)"
for loss in $DISTILL_LOSSES; do
  student="$(student_checkpoint 0.5 "$loss" "$teacher")"
  run_result 0.5 "$loss" constant "$student"
done

python summarize_dynamic_lr_ratio_l2.py \
  --root "$RUN_ROOT" \
  --base-sweep-root "$BASE_SWEEP_ROOT"

