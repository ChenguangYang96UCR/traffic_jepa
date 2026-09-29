# Masked Spatio-Temporal Graph JEPA

Standalone PyTorch pipeline for the proposed Fremont task:

- one graph snapshot per time step: `X_t = [93 sensors, 1 traffic value]`;
- a 12-step sample is `[12, 93, 1]`, not one sensor token containing 12 values;
- exactly `k` sensors are masked at every step;
- a spatial GNN processes each graph snapshot;
- a temporal Transformer processes the 12 representations of each sensor;
- an EMA target encoder supplies latent targets at masked sensor-time positions;
- a regression head predicts the actual value of every masked sensor-time position.

The saved MAE/MSE/RMSE are computed **only over masked positions**.

## Task definition

Default (bidirectional) mode is sensor-time imputation:

```text
[B, 12, 93, 1] -> mask k nodes at each step -> predict those 12*k values
```

The model may use observations before and after a masked step. Add `--causal` to
prevent temporal attention from seeing later steps. That variant is causal sensor
forecasting; do not report it in the same table as bidirectional imputation.

## Directory layout

```text
masked-st-jepa/
  masked_st_jepa/          model, data, graph and mask implementation
  scripts/                 ready-to-run Fremont and ablation scripts
  tests/                   synthetic end-to-end checks
  train.py                 pretrain, fine-tune and test entry point
  requirements.txt
```

No code outside this directory is imported, so the directory can be copied or
zipped and transferred to another server.

## Server setup

```bash
cd masked-st-jepa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -v
```

The internal package is deliberately named `masked_st_jepa`, so the project
directory itself may safely be renamed to `mstjepa` on the server without
shadowing the Python package.

If PyTorch/CUDA is already installed on the server, install only compatible
NumPy and pandas versions instead of replacing the existing PyTorch build.

## Run with existing Fremont incident windows

Expected data directory:

```text
Fremont/
  incident_train.npy
  incident_val.npy
  incident_test.npy
  adj_matrix.npy
```

From this project directory:

```bash
DATA_ROOT=/absolute/path/to/Fremont bash scripts/run_fremont.sh
```

The loader uses `sample['x_data'][:, :, traffic_feature]`; incident attributes
and `y_data` are deliberately not used. The existing split boundaries are kept.

Direct command:

```bash
python train.py \
  --data /absolute/path/to/Fremont \
  --data-format incident \
  --adjacency /absolute/path/to/Fremont/adj_matrix.npy \
  --graph-mode file \
  --traffic-feature 0 \
  --mask-mode random \
  --mask-k 9 \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --output runs/fremont_random_k9
```

Confirm what traffic feature index `0` represents in the raw XTraffic schema.

## Run with a continuous chronological series

Supported input is a `[time, sensor]` `.npy`, an `.npz` containing `data`, or a
wide CSV whose first nonnumeric column is a timestamp. Splitting is chronological
and happens before window construction, preventing windows from crossing split
boundaries. Normalization statistics come only from the training rows.

```bash
DATA_FILE=/path/to/fremont_flow.csv \
ADJACENCY=/path/to/adj_matrix.npy \
bash scripts/run_continuous.sh
```

If `ADJACENCY` is omitted, a top-k absolute-correlation graph is estimated using
training data only.

## Training stages and baselines

1. `pretrain`: optimize masked latent L1 loss; update target encoder by EMA.
2. `finetune`: optimize masked-value Smooth L1 plus `jepa_weight * latent_loss`.
3. `test`: use a fixed random mask seed and report masked-position metrics.

Useful controlled experiments:

```bash
# JEPA pretrain + full fine-tune (default)
--pretrain-epochs 20 --finetune-strategy full

# Frozen representation / regression-head probe
--pretrain-epochs 20 --finetune-strategy head

# Ordinary supervised ST-GNN baseline without JEPA pretraining
--pretrain-epochs 0 --jepa-weight 0
```

Run mask ablations with:

```bash
DATA_ROOT=/path/to/Fremont bash scripts/run_ablations.sh
```

Run the three primary controlled experiments and produce one summary table:

```bash
DATA_ROOT=/absolute/path/to/Fremont \
RUN_ROOT=runs/fremont_comparison \
bash scripts/run_comparison.sh
```

The script runs ST-GNN supervised, JEPA frozen probe, and JEPA full fine-tune
with the same graph, data splits, masks, seed, architecture, and fine-tuning
epochs. Results are written below `RUN_ROOT`, and `RUN_ROOT/summary.txt` contains
the final MAE/MSE/RMSE table.

Mask modes:

- `random`: independently sample k sensors at every step;
- `persistent`: hide the same k sensors for all 12 steps (sensor failure);
- `spatial_block`: hide connected graph neighborhoods.

## Outputs

Each run directory contains:

- `best_pretrain.pt`: best EMA-JEPA representation checkpoint;
- `best_finetune.pt`: best masked-value checkpoint;
- `test_metrics.json`: masked MAE, MSE, RMSE, losses and experiment metadata.

## Recommended first experiment

Use `k = 9` (about 10% of 93 sensors), three seeds, and compare:

| Experiment | pretrain epochs | JEPA weight | fine-tune |
|---|---:|---:|---|
| ST-GNN supervised | 0 | 0 | full |
| JEPA + frozen probe | 20 | 0 | head |
| JEPA + full fine-tune | 20 | 0.1 | full |

Then repeat for `k in {5, 9, 19, 28}` and all three mask modes. Never use
validation/test values to construct a correlation graph or normalization stats.

```bash
nohup env \
DATA_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/masked-st-jepa/dataset/Fremont \
RUN_ROOT=runs/fremont_comparison_k9 \
MASK_K=9 \
MASK_MODE=spatial_block \
PRETRAIN_EPOCHS=30 \
FINETUNE_EPOCHS=20 \
SEED=2026 \
bash scripts/run_comparison.sh \
> run_k19.log 2>&1 &
```