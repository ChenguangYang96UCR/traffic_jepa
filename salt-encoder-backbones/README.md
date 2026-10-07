# SALT as a transferable Transformer-encoder training framework

This package applies one fixed Oakland SALT Teacher to three forecasting
encoders without modifying their official core implementations:

| Backbone | Paper | Student latent used by SALT | Original downstream head |
|---|---|---|---|
| PatchTST | ICLR 2023 | temporal patch tokens `[B,P,N,D]` | shared flatten/linear head |
| STGformer | 2024 preprint | one pooled token per node `[B,1,N,D]` | linear horizon projection |
| STLAformer | TR-C 2026 | step/node tokens `[B,24,N,D]` | linear horizon projection |

The official sources and exact commits are recorded in
`UPSTREAM_COMMITS.json`. They are vendored under `upstream/`; all SALT-specific
logic lives in `salt_backbones/`.

## Controlled question

For each backbone the pipeline reports:

1. **Target scratch**: Berkeley forecasting from random initialization.
2. **SALT frozen**: Oakland Teacher -> Berkeley Student distillation, then train
   only the backbone's original forecasting head.
3. **SALT full FT**: the same distilled Student, then full Berkeley fine-tuning.
4. **Supervised transfer full FT**: supervised Oakland forecasting pretraining,
   followed by compatible encoder transfer and full Berkeley fine-tuning.

The fourth condition distinguishes a SALT effect from an ordinary supervised
transfer effect. All target checkpoints are selected using Berkeley validation
MAE; Berkeley test metrics are report-only.

## SALT boundary

During distillation the Student receives 12 observed history steps followed by
12 learned future-mask values. The frozen Teacher receives the corresponding
complete 24-step sequence. A temporary MLP Predictor maps Student width to
Teacher width and is discarded after distillation.

Teacher targets are aligned to each native encoder resolution:

- PatchTST: Teacher steps are mean-pooled inside the same length-4, stride-2
  temporal patches.
- STGformer: its official encoder collapses time before the forecasting head,
  so all 24 Teacher steps are pooled to one node token. This condition is not a
  step-level distillation objective and must be reported as such.
- STLAformer: all 24 step/node tokens are compared directly.

Calendar fields are intentionally disabled because the released city windows
do not contain verified timestamps. No future traffic label is passed into the
Student branch.

## Run

Install dependencies in the existing PyTorch environment:

```bash
pip install -r requirements.txt
```

Run one backbone:

```bash
export TEACHER_CHECKPOINT=/path/to/oakland_teacher/teacher/best_teacher.pt
export SOURCE_ROOT=/path/to/TopoJEPA/dataset/Oakland
export TARGET_ROOT=/path/to/TopoJEPA/dataset/Berkeley
export MODEL=patchtst
bash scripts/run_one.sh
```

Run all three sequentially:

```bash
export TEACHER_CHECKPOINT=/path/to/oakland_teacher/teacher/best_teacher.pt
export SOURCE_ROOT=/path/to/TopoJEPA/dataset/Oakland
export TARGET_ROOT=/path/to/TopoJEPA/dataset/Berkeley
nohup bash scripts/run_three.sh > run_salt_encoder_backbones.log 2>&1 &
```

To place different models on different GPUs, launch `run_one.sh` separately
with `DEVICE=cuda:0`, `DEVICE=cuda:1`, etc. Do not run several large models on
one GPU concurrently.

## Outputs

Each model directory contains independent checkpoints and metrics for scratch,
distillation, SALT frozen/full, supervised source, and supervised transfer.
The combined table is written to `summary.txt` after `run_three.sh` finishes.

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  DEVICE=cuda:0 \
  TEACHER_CHECKPOINT=/home/ADS/cyang314/ucr_work/traffic_jepa/staeformer-salt-future-focus/runs/oakland_teacher128_berkeley_sweep/teachers/blocks=8_ratio=0.5/teacher/best_teacher.pt \
  SOURCE_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Oakland \
  TARGET_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Berkeley \
  OUTPUT_ROOT=runs/oakland_teacher_berkeley_backbones \
  bash scripts/run_three.sh \
  > run_salt_encoder_backbones_gpu1.log 2>&1 &
```