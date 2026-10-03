#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OAKLAND_ROOT="${OAKLAND_ROOT:-../TopoJEPA/dataset/Oakland}"
BERKELEY_ROOT="${BERKELEY_ROOT:-../TopoJEPA/dataset/Berkeley}"
RUN_ROOT="${RUN_ROOT:-runs/oakland_teacher128_berkeley_sweep}"
MASK_BLOCKS_LIST="${MASK_BLOCKS_LIST:-4 8}"
FUTURE_RATIOS="${FUTURE_RATIOS:-0.5 0.75 1.0}"
FINETUNE_LRS="${FINETUNE_LRS:-0.0001 0.0005 0.001}"
SEED="${SEED:-2026}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-50}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATIENCE="${PATIENCE:-10}"

for root in "$OAKLAND_ROOT" "$BERKELEY_ROOT"; do
  for split in train val test; do
    file="$root/incident_${split}.npy"
    if [[ ! -f "$file" ]]; then
      echo "ERROR: dataset split not found: $file" >&2
      exit 2
    fi
  done
done

mkdir -p "$RUN_ROOT"

for mask_blocks in $MASK_BLOCKS_LIST; do
  for future_ratio in $FUTURE_RATIOS; do
    tag="blocks=${mask_blocks}_ratio=${future_ratio}"
    teacher_output="$RUN_ROOT/teachers/$tag"
    teacher_checkpoint="$teacher_output/teacher/best_teacher.pt"
    student_output="$RUN_ROOT/students/$tag"
    student_checkpoint="$student_output/student/best_student.pt"

    echo "=== Teacher: mask_blocks=$mask_blocks future_ratio=$future_ratio ==="
    if [[ ! -f "$teacher_checkpoint" ]]; then
      python run.py \
        --mode teacher \
        --teacher-data "$OAKLAND_ROOT" \
        --teacher-input-dim 64 \
        --teacher-step-dim 64 \
        --teacher-sensor-dim 0 \
        --teacher-ff-dim 256 \
        --teacher-heads 4 \
        --teacher-layers 3 \
        --teacher-mask-policy future_biased \
        --teacher-future-block-ratio "$future_ratio" \
        --mask-blocks "$mask_blocks" \
        --input-steps 12 \
        --pred-steps 12 \
        --teacher-epochs "$TEACHER_EPOCHS" \
        --teacher-lr 0.001 \
        --batch-size "$BATCH_SIZE" \
        --patience "$PATIENCE" \
        --seed "$SEED" \
        --output "$teacher_output"
    else
      echo "Reusing $teacher_checkpoint"
    fi

    echo "=== Berkeley Student: all-step latent loss ==="
    if [[ ! -f "$student_checkpoint" ]]; then
      python run.py \
        --mode student \
        --teacher-checkpoint "$teacher_checkpoint" \
        --expected-teacher-dim 128 \
        --expected-teacher-mask-policy future_biased \
        --expected-teacher-future-block-ratio "$future_ratio" \
        --expected-teacher-mask-blocks "$mask_blocks" \
        --student-data "$BERKELEY_ROOT" \
        --student-distill-scope all \
        --input-steps 12 \
        --pred-steps 12 \
        --student-epochs "$STUDENT_EPOCHS" \
        --student-lr 0.0001 \
        --batch-size "$BATCH_SIZE" \
        --patience "$PATIENCE" \
        --seed "$SEED" \
        --output "$student_output"
    else
      echo "Reusing $student_checkpoint"
    fi

    # The scratch result is independent of the SALT hyperparameters. Run it once.
    if [[ ! -f "$RUN_ROOT/baseline/all_results.json" ]]; then
      echo "=== Berkeley scratch baseline ==="
      python run.py \
        --pipeline in_domain \
        --mode downstream \
        --student-checkpoint "$student_checkpoint" \
        --student-data "$BERKELEY_ROOT" \
        --target-data "$BERKELEY_ROOT" \
        --student-distill-scope all \
        --downstream-strategies scratch \
        --input-steps 12 \
        --pred-steps 12 \
        --finetune-epochs "$FINETUNE_EPOCHS" \
        --finetune-lr 0.001 \
        --batch-size "$BATCH_SIZE" \
        --patience "$PATIENCE" \
        --seed "$SEED" \
        --output "$RUN_ROOT/baseline"
    fi

    for finetune_lr in $FINETUNE_LRS; do
      result_output="$RUN_ROOT/results/$tag/lr=$finetune_lr"
      if [[ -f "$result_output/all_results.json" ]]; then
        echo "Reusing completed result: $result_output"
        continue
      fi
      echo "=== Full FT: blocks=$mask_blocks ratio=$future_ratio lr=$finetune_lr ==="
      python run.py \
        --pipeline in_domain \
        --mode downstream \
        --experiment-label "Oakland Teacher -> Berkeley Student | all-step | blocks=$mask_blocks ratio=$future_ratio lr=$finetune_lr" \
        --student-checkpoint "$student_checkpoint" \
        --student-data "$BERKELEY_ROOT" \
        --target-data "$BERKELEY_ROOT" \
        --student-distill-scope all \
        --downstream-strategies full \
        --input-steps 12 \
        --pred-steps 12 \
        --finetune-epochs "$FINETUNE_EPOCHS" \
        --finetune-lr "$finetune_lr" \
        --encoder-lr-scale 0.1 \
        --batch-size "$BATCH_SIZE" \
        --patience "$PATIENCE" \
        --seed "$SEED" \
        --output "$result_output"
    done
  done
done

python summarize_salt_sweep.py --root "$RUN_ROOT"
