# Oakland multi-city transfer benchmark

## SALT-only backbone ablation across three counties

`scripts/run_salt_three_counties.sh` now trains **only** the SALT variants of
the ten backbones used in the paper table:

`PDFormer, FlashST, TESTAM, PatchSTG, SC-JEPA, TS-JEPA, DCRNN, AGCRN,
SDGFormer, DSTFGCN`.

For every county, horizon, and backbone, SALT trains an independent Teacher
whose encoder has the same backbone structure as its Student. Each pair uses
8 topology-aware spatiotemporal mask blocks and a 0.5 future-block ratio. SALT
trains the backbone-specific Student encoder plus a temporary latent predictor,
discards that predictor, transfers the encoder, and performs full fine-tuning
on each target city. Teacher checkpoints are reusable across target cities but
never shared across different backbone families. The established `Ours` and
ordinary transfer rows are not rerun.

```bash
nohup env \
  DATA_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset \
  DEVICE=cuda:1 \
  BASELINE_CSVS=/path/alameda/results.csv:/path/orange/results.csv:/path/contra/results.csv \
  bash scripts/run_salt_three_counties.sh \
  > run_salt_three_counties.log 2>&1 &
```

The new runs are written to `salt_results.csv`. The summarizer joins them with
the read-only files in `BASELINE_CSVS` and produces
`combined_existing_and_salt_results.csv`; missing old rows remain blank rather
than being retrained.

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
| SDGFormer (2026) / DSTFGCN (2026) | supervised Oakland forecasting with the official released core | full fine-tuning |

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

The benchmark uses its local `salt_backbones` package when present. A sibling
`salt-jepa-eight-backbones/salt_backbones` checkout is supported only as a
compatibility fallback. SC-JEPA uses `sc-jepa-official-traffic`, while TS-JEPA
uses the pinned official source downloaded into `upstream/TS_JEPA`. TS-JEPA
needs only the existing PyTorch and NumPy stack. No additional package is
needed for DCRNN/AGCRN.

SDGFormer is pinned at upstream commit
`7c441378a8ef21f74de80817a60679887516faf5`; DSTFGCN is pinned at
`85d8659ad04782611356a1a959dcbca07025a3b1`. Their official model cores are in
`upstream/`. The adapters use traffic feature 0 and the released adjacency.
The SDGFormer adapter loads its released model file through a local compatibility
shim; **do not install `libcity` from pip** (there is no required PyPI package).
DSTFGCN's absolute time lookup is set to one shared ID because the released
city windows do not contain verified absolute timestamps; its relative
12-step temporal graph remains active.

Run only the two recent additions (all cities and horizons):

```bash
nohup env DEVICE=cuda:1 DATA_ROOT=/path/to/TopoJEPA/dataset \
  bash scripts/run_recent.sh > run_sdgformer_dstfgcn.log 2>&1 &
```

## Orange County: Irvine source transfer

The Orange County experiment first derives node-aligned city views from the
county-level `Orange/incident_{train,val,test}.npy`, `sensors.csv`, and
`adj_matrix.npy`. It then trains each source method on Irvine and fully
fine-tunes on Laguna Niguel, Newport Beach, Westminster, Fullerton, Buena Park,
and Mission Viejo. For Ours, both the masked Teacher and distilled Student are
trained on Irvine before transfer.

```bash
nohup env \
  DATA_ROOT=/path/to/TopoJEPA/dataset \
  ORANGE_ROOT=/path/to/TopoJEPA/dataset/Orange \
  DEVICE=cuda:1 \
  bash scripts/run_irvine_orange.sh \
  > run_irvine_orange.log 2>&1 &
```

The preparation stage is idempotent with `--reuse-existing` and writes a
`city_view_manifest.json` containing the original county sensor indices for
each derived city.

### Ten-percent target fine-tuning

`run_irvine_orange_fewshot10.sh` reuses the full Irvine source checkpoints and
fine-tunes every method on the same deterministic random 10% subset of each
target-city training split. Validation and test splits remain complete.

### Traffic-SALT mask ablation

The mask ablation keeps the number of blocks, time-span distribution,
sensor-count distribution, and future-block ratio fixed. `random_st` samples
the sensors independently; `topology_st` starts from a uniformly sampled road
sensor and expands a size-controlled connected region using the symmetrized
weighted adjacency graph.

```bash
nohup env DEVICE=cuda:1 \
  bash scripts/run_irvine_mask_ablation.sh \
  > run_irvine_mask_ablation.log 2>&1 &
```

Set `TARGET_TRAIN_FRACTION=0.1` to run the same mask comparison under the 10%
few-shot target protocol. The script writes `mask_ablation_results.csv` and
`mask_ablation_table.tex` under its output root.

### Single-city deployment-efficiency benchmark

The efficiency benchmark fixes Irvine as the source, Laguna Niguel as the
target, horizon 12, and seed 2026. It trains or reuses the same transfer
checkpoints and profiles only the deployed encoder plus forecasting head.
Offline Teachers and latent predictors are not included in deployment latency
or deployment parameter counts.

```bash
nohup env DEVICE=cuda:1 \
  bash scripts/run_irvine_single_city_efficiency.sh \
  > run_irvine_single_city_efficiency.log 2>&1 &
```

The profiler writes `efficiency/efficiency.csv`, `efficiency.json`, and an
Overleaf-ready `efficiency_table.tex`. CUDA timing uses 50 warm-up iterations
and 200 synchronized measurements at batch sizes 1 and 32 by default.

## SALT across backbones and counties

`run_salt_three_counties.sh` evaluates SALT on the ten paper-table backbones.
The controlled
protocol uses one source and two targets per county:

- Alameda: Oakland -> Berkeley, Dublin
- Orange: Irvine -> Laguna Niguel, Fullerton
- Contra Costa: Richmond -> Hercules, Pleasant Hill

For every backbone and horizon, it trains that backbone's reconstruction
Teacher, its same-family SALT Student, and the target-city full fine-tuning
runs. Existing comparison rows are loaded from CSV files and are not retrained.
Each Teacher/Student pair is trained once and reused across both target cities;
it is not reused by another backbone. Every Ours/Traffic-SALT run uses
`topology_st` with eight blocks and a 50:50 history/future block allocation.

```bash
nohup env DEVICE=cuda:1 \
  bash scripts/run_salt_three_counties.sh \
  > run_salt_three_counties.log 2>&1 &
```
```bash
nohup env \
  DEVICE=cuda:1 \
  bash scripts/run_salt_three_counties.sh \
  > run_salt_three_counties_backbone_teacher.log 2>&1 &
```


Set `COUNTY_FILTER=alameda`, `orange`, or `contra_costa` to run one county, or
set `HORIZONS=12` for a one-horizon smoke test. Results are resumable and are
summarized in `salt_vs_existing_summary.csv` and
`salt_vs_existing_summary.tex`.

## Three-county masking ablation

The controlled Ours-only masking study covers Alameda (Oakland to Berkeley and
Dublin), Orange (Irvine to Laguna Niguel and Fullerton), and Contra Costa
(Richmond to Hercules and Pleasant Hill), at horizons 6/9/12. It evaluates:

- random spatiotemporal blocks at history:future ratios 1:9, 3:7, 5:5, 8:2;
- topology-connected blocks at the same four ratios; and
- uniform random points with no prescribed history/future allocation.

Eight mask blocks and their size distribution stay fixed. Since eight blocks
cannot encode every tenth-based ratio exactly, each block independently selects
the future region with probability 0.9, 0.7, 0.5, or 0.2; the requested ratio
therefore holds in expectation over samples. Random point first matches each
sample's block-mask count, then redraws that many isolated locations uniformly
over the full space-time tensor.

```bash
nohup env \
  DATA_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset \
  DEVICE=cuda:1 \
  bash scripts/run_mask_ablation_three_counties.sh \
  > run_mask_ablation_three_counties.log 2>&1 &
```

The run is resumable. Its main outputs are `mask_ablation_results.csv`,
`mask_ablation_city_horizon_average.csv`, `mask_ablation_overall_ranking.csv`,
and `mask_ablation_overall_table.tex`.
