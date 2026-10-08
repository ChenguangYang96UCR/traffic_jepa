# Shared-Teacher SALT for eight traffic Transformer backbones

This is a separate experiment package; it does not modify
`salt-jepa-eight-backbones`. It compares PDFormer, FlashST, PatchSTG, TESTAM,
PatchTST, STAEformer, STGormer, and TSFormer using one common SALT Teacher.
Official backbone code is vendored under `upstream/` and commits are recorded
in `UPSTREAM_COMMITS.json`.

## Pipeline

For a fixed source city, horizon, and seed, the Teacher is trained exactly once:

```text
Oakland complete history+future tensor
  -> shared masked reconstruction Teacher
  -> freeze one shared Teacher checkpoint
  -> independently distill each Student backbone on Oakland
  -> discard each temporary latent predictor
  -> transfer the Student encoder to Berkeley
  -> full fine-tune Student encoder + native forecast head
  -> Berkeley test
```

Every backbone therefore receives identical Teacher targets. The Teacher is not
an instance of any tested forecasting backbone.

The package reports four target-test methods:

1. `target scratch`
2. `supervised source -> target full FT`
3. `Shared SALT -> full FT`
4. `JEPA -> full FT`

Frozen evaluation is excluded. Target checkpoints are selected only by target
validation MAE; test metrics are report-only.

## Information represented by the shared Teacher

The Teacher consumes `[batch, history+future, sensor, traffic_channel]` and
always returns `[batch, history+future, sensor, 128]`. Its input representation
contains:

- raw traffic values;
- first temporal differences;
- graph-neighbour traffic averages;
- an explicit mask indicator and learned mask token;
- learned relative step positions;
- a history/future segment embedding;
- normalized adjacency graph diffusion;
- deterministic degree and Laplacian positional features.

Alternating temporal attention, spatial attention, and graph diffusion preserve
the full sensor-time grid. The Teacher uses no learned city-specific node table.
Because the released arrays have no verified timestamps, time-of-day and
day-of-week embeddings are deliberately not fabricated. Incident features are
not loaded.

Patch-based Students call their own `align_teacher()` operation to pool the
Teacher's raw-step latents to the Student token grid. A Student-specific MLP
predictor maps each Student latent width to the common Teacher width. Predictors
and the Teacher reconstruction head are discarded before forecasting.

## Data

Each city directory must contain `incident_train.npy`, `incident_val.npy`,
`incident_test.npy`, and `adj_matrix.npy`. Only `x_data`, `y_data`, and traffic
feature 0 are used. The supported protocol is 12 history steps and a 6, 9, or
12 step horizon.

## Environment

Keep the server's CUDA-compatible PyTorch installation. Do not allow pip to
replace it:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install -r requirements.txt
```

## Train the shared Teacher once

```bash
cd salt-shared-teacher-backbones
export SOURCE_ROOT=/path/to/TopoJEPA/dataset/Oakland
export OUTPUT_ROOT=runs/shared_teacher_backbones
export HORIZON=12
export SEED=2026
export DEVICE=cuda:1
bash scripts/train_shared_teacher.sh
```

```bash
cd salt-shared-teacher-backbones
export SOURCE_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Oakland
export OUTPUT_ROOT=runs/shared_teacher_backbones
export HORIZON=12
export SEED=2026
export DEVICE=cuda:1
bash scripts/train_shared_teacher.sh
```

The checkpoint is written to:

```text
runs/shared_teacher_backbones/shared_teacher/horizon_12/seed_2026/best_teacher.pt
```

It records the architecture, horizon, source graph, mask settings, dimensions,
and reconstruction metrics. All eight Students reuse this exact file.

## Run one backbone

```bash
export MODEL=staeformer
export SOURCE_ROOT=/path/to/TopoJEPA/dataset/Oakland
export TARGET_ROOT=/path/to/TopoJEPA/dataset/Berkeley
export OUTPUT_ROOT=runs/shared_teacher_backbones
export HORIZON=12
export SEED=2026
export DEVICE=cuda:1
bash scripts/run_one.sh
```

To use a checkpoint elsewhere, set `TEACHER_CHECKPOINT=/absolute/path/best_teacher.pt`.
The loader verifies the horizon, masking configuration, node count, and source
adjacency before distillation.

## Run all eight backbones

The runner automatically trains one missing Teacher per seed, then reuses it:

```bash
export SOURCE_ROOT=/path/to/TopoJEPA/dataset/Oakland
export TARGET_ROOT=/path/to/TopoJEPA/dataset/Berkeley
export OUTPUT_ROOT=runs/shared_teacher_backbones
export DEVICE=cuda:1
export SEEDS="2024 2025 2026"
nohup bash scripts/run_eight.sh > run_shared_teacher_eight.log 2>&1 &
```

`summary_mean_std.csv` contains MAE/MSE/RMSE mean and sample standard deviation
over the three seeds.

## Horizon sweep at seed 2026

```bash
export SOURCE_ROOT=/path/to/TopoJEPA/dataset/Oakland
export TARGET_ROOT=/path/to/TopoJEPA/dataset/Berkeley
export OUTPUT_ROOT=runs/shared_teacher_horizon_sweep_seed2026
export DEVICE=cuda:1
nohup bash scripts/run_horizons.sh > run_shared_teacher_horizons.log 2>&1 &
```

This creates exactly one Teacher for each horizon, not one per backbone.

## Default controlled settings

- Teacher width 128, 3 blocks, 8 heads, FFN width 512
- 8 spatiotemporal mask blocks
- future block ratio 0.50
- time extent 2--6 steps
- sensor extent 10--30%
- L2 Student latent loss
- full target fine-tuning LR `1e-3`
- Student encoder LR scale `0.1`
- default device `cuda:1`

The original matched-Teacher package should be retained as a separate ablation:
shared Teacher tests architecture decoupling, while matched Teachers test
backbone-specific target representations.

```bash
nohup env \
SOURCE_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Oakland \
TARGET_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Berkeley \
OUTPUT_ROOT=runs/shared_teacher_horizon_sweep_seed2026 \
DEVICE=cuda:2 \
SEED=2026 \
HORIZONS="6 9 12" \
bash scripts/run_horizons.sh \
> run_shared_teacher_horizons.log 2>&1 &
```