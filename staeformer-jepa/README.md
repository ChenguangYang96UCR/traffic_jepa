# STAEformer-JEPA: future-flow forecasting

Standalone PyTorch package for testing whether JEPA representation pretraining
improves the standard STAEformer downstream task.

## What remains unchanged

The downstream task is direct future traffic-flow forecasting:

```text
past 12 steps x 93 sensors -> future 3 steps x 93 sensors
```

There is no random sensor mask, no missing-value reconstruction, and no JEPA
loss during downstream fine-tuning. The forecasting backbone retains the core
published STAEformer design: traffic/time/adaptive embeddings, temporal vanilla
Transformer layers, spatial vanilla Transformer layers, and a mixed projection
from all historical representations to all future values.

## JEPA pretraining

The online STAEformer encoder sees only the 12 historical steps. Future queries
predict the latent representations of all `3 x 93` future positions. An EMA
target STAEformer encoder independently encodes the true 3-step future block and
supplies stop-gradient latent targets. Both branches share the standard learned
12-position adaptive-embedding table; the shorter target block uses its first
three relative positions. Pretraining uses latent L1 only: it never regresses
traffic flow.

After pretraining, the target encoder and JEPA predictor are discarded. The
online encoder initializes a standard STAEformer forecast model.

## Experiments produced by one command

1. STAEformer trained from scratch.
2. JEPA-pretrained encoder frozen; train the original forecast head only.
3. JEPA-pretrained encoder fully fine-tuned with forecast loss only.

## Server installation

```bash
cd staeformer-jepa
pip install -e . --no-deps
python -m unittest discover -s tests -v
```

The package only requires a compatible PyTorch and NumPy. `--no-deps` avoids
replacing the server's existing CUDA-enabled PyTorch build.

## Fremont data

Expected directory:

```text
Fremont/
  incident_train.npy
  incident_val.npy
  incident_test.npy
```

Each object-array sample must contain:

```text
x_data: [12, 93, >=3]
y_data: [12, 93, >=3]
```

By default channel 0 is traffic, channel 1 is normalized time-of-day, and
channel 2 is day-of-week in `[0,6]`. Only `y_data[:3, :, 0]` is the downstream
target. Incident metadata is not loaded.

## Run the complete comparison

```bash
DATA_ROOT=/absolute/path/to/TopoJEPA/dataset/Fremont \
OUTPUT_DIR=runs/fremont_12_to_3 \
bash scripts/run_fremont_12_to_3.sh
```

For a background run:

```bash
nohup env \
DATA_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Fremont \
OUTPUT_DIR=runs/fremont_12_to_3 \
bash scripts/run_fremont_12_to_3.sh \
> run_fremont_12_to_3.log 2>&1 &
```

## Outputs

```text
runs/fremont_12_to_3/
  pretrain/best_jepa.pt
  scratch/best_forecast.pt
  scratch/test_metrics.json
  frozen/best_forecast.pt
  frozen/test_metrics.json
  full/best_forecast.pt
  full/test_metrics.json
  all_results.json
  summary.txt
```

Metrics include overall MAE/MSE/RMSE and separate values for future steps 1, 2,
and 3. With 5-minute sampling these are 5-, 10-, and 15-minute forecasts.

## Important interpretation

The released incident windows are already stored/normalized, so reported values
remain in stored units unless the original inverse-scaling statistics are known.
The same incident windows may overlap. This experiment therefore evaluates the
released split, not a newly reconstructed continuous chronological benchmark.

## Attribution

The STAEformer architecture follows Liu et al., *Spatio-Temporal Adaptive
Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting*, CIKM 2023,
and its official repository: https://github.com/XDZhelheim/STAEformer. This
package is a clean implementation for the present experiment and does not vendor
the official source tree or datasets.
