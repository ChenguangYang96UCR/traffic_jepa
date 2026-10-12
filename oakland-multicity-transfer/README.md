# Oakland multi-city transfer benchmark

This package runs one controlled transfer-learning table:

- source: **Oakland**;
- targets: **Berkeley, Dublin, Newark, San Leandro, Pleasanton, Livermore**;
- history: 12 traffic steps;
- horizons: 6, 9, and 12;
- seed: 2026;
- checkpoint selection: lowest target-city validation MAE;
- report: target-city test MAE, MSE, and RMSE;
- transfer policy: full fine-tuning (no frozen-probe rows).

## Methods

| Table name | Source training | Target training |
|---|---|---|
| Ours (STAEformer+SALT) | masked reconstruction Teacher + STAEformer Student distillation on Oakland | full fine-tuning |
| PDFormer / FlashST / TESTAM / PatchSTG / STAEformer | supervised Oakland forecasting | full fine-tuning |
| SC-JEPA | official SC-JEPA self-supervised pretraining on Oakland | full fine-tuning with a fresh horizon-specific traffic head |
| TS-JEPA | official TS-JEPA temporal-patch JEPA pretraining on Oakland | full fine-tuning with a fresh horizon-specific traffic head |
| STAEformer-JEPA | EMA-target JEPA pretraining on Oakland | full fine-tuning |
| DCRNN / AGCRN | supervised Oakland forecasting | full fine-tuning |

The source checkpoint is trained once per `(method, horizon, seed)` and reused
for all six target cities. SC-JEPA's official representation task always uses
four 3-step target patches (12 target steps), so its single Oakland checkpoint
is shared by the 6/9/12-step downstream heads. City-specific sensor embeddings,
graph buffers, and forecast heads are reinitialized when node counts differ.
TS-JEPA treats each sensor as a time series with four non-overlapping 3-step
patches. It uses the paper's 70% patch mask and EMA momentum 0.998. Because its
temporal encoder is node-independent and consumes only the 12-step history, one
Oakland encoder checkpoint is shared by all cities and horizons; each horizon
gets a fresh linear forecast head before full fine-tuning.

Only traffic feature 0 is loaded. Incident features are not used.

## Dataset layout

Each city folder must contain:

```text
TopoJEPA/dataset/<City>/
  incident_train.npy
  incident_val.npy
  incident_test.npy
  adj_matrix.npy
```

`San Leandro`, `San_Leandro`, and `SanLeandro` directory spellings are detected.

## Run all experiments

Fetch the pinned official TS-JEPA source once (commit
`830ee5c3a7d4e85623bd5a07992732be7fda9bc2`):

```bash
bash scripts/fetch_ts_jepa.sh
```

`run_all.sh` also performs this check automatically.

From this directory:

```bash
nohup env \
  DATA_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset \
  DEVICE=cuda:1 \
  bash scripts/run_all.sh \
  > run_oakland_multicity.log 2>&1 &
```

To run or resume one cell:

```bash
env DEVICE=cuda:1 DATA_ROOT=/path/to/TopoJEPA/dataset \
  bash scripts/run_one.sh ts_jepa Berkeley 6
```

Completed target runs and source checkpoints are automatically reused. Pass
`--force` directly to `run.py` only when target fine-tuning should be repeated.

To add only TS-JEPA to an existing completed benchmark table:

```bash
nohup env DEVICE=cuda:1 DATA_ROOT=/path/to/TopoJEPA/dataset \
  bash scripts/run_ts_jepa.sh > run_ts_jepa.log 2>&1 &
```

## Outputs

```text
runs/oakland_multicity_seed2026/
  source/                         # reusable Oakland checkpoints
  targets/<city>/h<horizon>/...  # transfer reports and target checkpoints
  results.csv                     # long-form results
  paper_table_mae_mse.csv         # city x horizon x method table
  paper_table_mae_mse.tex         # LaTeX table matching the requested layout
```

The `Avg` row is the arithmetic mean of the three horizon-level metrics. Test
metrics are report-only; test performance is never used to select checkpoints.

## Dependencies

The package intentionally imports the already vendored official adapters from
the sibling folders `salt-jepa-eight-backbones` and
`sc-jepa-official-traffic`, and the pinned official source downloaded into
`upstream/TS_JEPA`. TS-JEPA needs only the existing PyTorch and NumPy stack.
No additional package is needed for DCRNN/AGCRN.
