# STAEformer-SALT traffic forecasting

This is the clean SALT-only implementation. It is intentionally separate from
the earlier EMA-based latent-prediction baseline.

## Pipeline

1. Train a small STAEformer teacher using masked spatio-temporal block
   reconstruction.
2. Freeze that teacher permanently.
3. Train a larger STAEformer student and asymmetric predictor to match the
   teacher's future latent representations.
4. Discard the teacher, reconstruction decoder, and predictor.
5. Attach a flow forecast head to the student and evaluate frozen probing or
   full fine-tuning against a from-scratch STAEformer baseline.

The downstream task is past 12 traffic steps to future 12 traffic steps. No
incident, time-of-day, or day-of-week feature is used.

## Install and test

```bash
cd staeformer-salt
pip install -e . --no-deps
python -m unittest discover -s tests -v
```

## Fremont-only, no transfer

Teacher training, student distillation, downstream fitting, validation, and
testing all use Fremont, while preserving the provided train/validation/test
boundaries:

```bash
DATA_ROOT=/absolute/path/to/Fremont \
OUTPUT_DIR=runs/fremont_salt \
bash scripts/run_fremont.sh
```

## Oakland to Fremont transfer

Teacher and student are trained on Oakland. The student is then transferred to
Fremont. The source sensor embedding is not transferred because sensor count
and identity differ.

```bash
SOURCE_ROOT=/absolute/path/to/Oakland \
TARGET_ROOT=/absolute/path/to/Fremont \
OUTPUT_DIR=runs/oakland_to_fremont_salt \
bash scripts/run_oakland_to_fremont.sh
```

Both commands report:

```text
STAEformer scratch
SALT student -> frozen
SALT student -> full FT
```

Outputs include `teacher/best_teacher.pt`, `student/best_student.pt`, the three
downstream checkpoints, `all_results.json`, and `summary.txt`.
