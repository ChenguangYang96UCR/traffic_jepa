# Eight traffic-Transformer backbones: Scratch vs Transfer vs SALT vs JEPA

This package runs matched `12 history -> 6/9/12 future` forecasting protocols on
PDFormer, FlashST, PatchSTG, TESTAM, PatchTST, STAEformer, STGormer, and TSFormer
(the encoder from STEP). Official model code is vendored under `upstream/`;
exact source commits are in `UPSTREAM_COMMITS.json`.

## Compared methods

For every backbone, the pipeline reports exactly four target-test results:

1. `target scratch`: random initialization trained on Berkeley.
2. `supervised source -> target full FT`: supervised Oakland pretraining,
   compatible encoder transfer, then full Berkeley fine-tuning.
3. `SALT -> full FT`: a backbone-specific Teacher is trained on Oakland with
   future-biased block-mask reconstruction; an architecture-matched Oakland
   Student is distilled from it, transferred, and fully fine-tuned on Berkeley.
4. `JEPA -> full FT`: an online encoder predicts its Berkeley EMA-target latent;
   the EMA encoder is then fully fine-tuned with the native forecast head.

Frozen-probe evaluation is intentionally excluded. Target checkpoints are
selected by Berkeley validation MAE; test metrics are report-only. SALT/JEPA
predictors are temporary pretraining modules and are discarded downstream.

All methods use the same traffic field, split files, history/horizon, seed set,
optimizer family, batch size, stopping rule, and native backbone head. Incident
features are not loaded. Calendar embeddings are disabled because the released
arrays do not contain verified timestamps.

## Data and graph inputs

Each city directory must contain `incident_train.npy`, `incident_val.npy`,
`incident_test.npy`, and `adj_matrix.npy`. Samples must provide `x_data` and
`y_data`; only traffic feature 0 is used. Source and target may have different
node counts. City-specific node embeddings, graph buffers, pattern keys, and
forecast heads are rebuilt instead of transferred.

## Environment

Keep the server's working CUDA-compatible PyTorch installation; do not let pip
replace it:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install -r requirements.txt
```

NumPy is pinned to 1.23.5 for the existing traffic environment. Use a clean
Conda environment if another workload requires TensorFlow with NumPy 2.x.

## Run one backbone on GPU 1

```bash
cd salt-jepa-eight-backbones
export MODEL=staeformer
export SOURCE_ROOT=/path/to/TopoJEPA/dataset/Oakland
export TARGET_ROOT=/path/to/TopoJEPA/dataset/Berkeley
export OUTPUT_ROOT=runs/matched_teacher_backbones
export DEVICE=cuda:1
export SEED=2026
bash scripts/run_one.sh
```

`SOURCE_ADJ` and `TARGET_ADJ` default to `adj_matrix.npy` in each city root.

The first run writes the matched Teacher to
`<output>/teacher/best_teacher.pt`. To reuse it for another Student run, set:

```bash
export TEACHER_CHECKPOINT=/path/to/teacher/best_teacher.pt
bash scripts/run_one.sh
```

The checkpoint records and validates its backbone, horizon, latent width, mask
blocks, future ratio, node count, and original source dataset. When the Student
dataset has a different node count or city path, only compatible encoder
weights are loaded; the new city's graph and city-specific embeddings are
rebuilt.

## Run all eight sequentially on GPU 1

```bash
export SOURCE_ROOT=/path/to/TopoJEPA/dataset/Oakland
export TARGET_ROOT=/path/to/TopoJEPA/dataset/Berkeley
export OUTPUT_ROOT=runs/matched_teacher_backbones
export DEVICE=cuda:1
export SEEDS="2024 2025 2026"
nohup bash scripts/run_eight.sh > run_eight_backbones.log 2>&1 &
```

`BACKBONES` and `SEEDS` may select a subset when resuming an interrupted run.
Set `EXPECTED_SEEDS=3` so final aggregation validates all three seeds already
accumulated under `OUTPUT_ROOT`.

The example protocol runs every method independently with seeds 2024, 2025,
and 2026; `SEEDS` may be changed to any three distinct integers. Checkpoints
and raw rows are stored under
`OUTPUT_ROOT/<backbone>/seed_<seed>/`. The final file
`OUTPUT_ROOT/summary_mean_std.csv` reports test MAE, MSE, and RMSE as mean plus
or minus sample standard deviation (`ddof=1`) across the three seeds.

For a one-seed smoke test, set `SEEDS=2026 ALLOW_NONTHREE_SEEDS=1`, together
with `TEACHER_EPOCHS=1 STUDENT_EPOCHS=1 FORECAST_EPOCHS=1 SOURCE_EPOCHS=1
JEPA_EPOCHS=1`.
Regenerate the aggregate table with:

```bash
python summarize.py --root runs/matched_teacher_backbones --expected-seeds 3
```

## Adaptation boundary

The framework exposes each official encoder through
`encode_sequence([B,12+H,N,1]) -> latent` and
`forward([B,12,N,1]) -> [B,H,N,1]`, where `H` is 6, 9, or 12.
Future inputs are learned mask tokens.
Each SALT Teacher uses the same encoder class and latent width as its Student.
The Teacher reconstructs only masked traffic entries; patch-encoder latent
sequences are interpolated to raw time resolution only inside the temporary
reconstruction decoder. The Student predictor and Teacher decoder are both
discarded downstream. Graph
backbones receive the city adjacency; PDFormer and FlashST also derive semantic
profiles and pattern keys from the training split.

The central official encoder operations are retained, while input/output
wrappers are adapted to the released `x_data/y_data` windows. Report this as a matched
backbone adaptation, not a reproduction of every paper's benchmark setup.

## Single-seed horizon sweep

To compare all four methods at 6-, 9-, and 12-step horizons while keeping
history length 12 and seed 2026 fixed:

```bash
export SOURCE_ROOT=/path/to/TopoJEPA/dataset/Oakland
export TARGET_ROOT=/path/to/TopoJEPA/dataset/Berkeley
export OUTPUT_ROOT=runs/matched_teacher_horizon_sweep_seed2026
export DEVICE=cuda:2
nohup bash scripts/run_horizons.sh > run_horizons_seed2026.log 2>&1 &
```

Every backbone and horizon trains an independent Oakland Teacher and Student,
then transfers the distilled Student encoder to Berkeley. Teacher masking is
fixed to 8 blocks with future-block-ratio 0.50 unless overridden through
`MASK_BLOCKS` and `TEACHER_FUTURE_BLOCK_RATIO`.

To reuse a previously completed set of matched Teachers, point `TEACHER_ROOT`
at the earlier sweep root. The runner resolves each checkpoint as
`$TEACHER_ROOT/<backbone>/horizon_<H>/seed_2026/teacher/best_teacher.pt` and
skips Teacher training:

```bash
export TEACHER_ROOT=/path/to/runs/matched_teacher_horizon_sweep_seed2026
export OUTPUT_ROOT=runs/new_student_dataset_seed2026
nohup bash scripts/run_horizons.sh > reuse_teachers.log 2>&1 &
```
Outputs are separated under
`<backbone>/horizon_<6|9|12>/seed_2026/`. The final comparison is written to
`horizon_summary_seed2026.csv`. This is a single-seed ablation, so it reports
raw MAE/MSE/RMSE rather than mean and standard deviation.


```bash
export SOURCE_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Oakland
export TARGET_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Berkeley
export OUTPUT_ROOT=runs/matched_teacher_horizon_sweep_seed2026
export DEVICE=cuda:2
export SEED=2026

nohup bash scripts/run_horizons.sh \
  > run_horizons_seed2026__matched_teachers.log 2>&1 &

```