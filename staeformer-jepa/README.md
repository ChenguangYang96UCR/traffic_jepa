# STAEformer-JEPA traffic forecasting

Standalone PyTorch package for causal future-block JEPA pretraining followed by
STAEformer future-flow forecasting.

## Task and architecture

The downstream task remains:

```text
past 12 steps x N sensors -> future 12 steps x N sensors
```

During downstream forecasting, the 12 unknown future positions are appended as
mask tokens; no future traffic value is supplied to the model. Because the
released windows do not contain trustworthy absolute timestamps, the model
does not use time-of-day or day-of-week embeddings. It uses traffic values and:

```text
relative_step_embedding[T,D] + city_sensor_embedding[N,D]
```

This factorizes the original joint adaptive table. The relative step embedding
and all Transformer weights transfer between cities; each city learns its own
sensor embedding. Oakland and Fremont may therefore have different node counts
without inventing a node-to-node mapping.

The encoder consists of traffic projection, relative-step embedding,
city-specific sensor embedding, temporal self-attention, spatial
self-attention, feed-forward layers, and normalization.

## JEPA pretraining

Each source sample is a 24-step sequence made from 12 historical and 12 future
traffic steps. In the default `future` mode, the online branch always receives
the first 12 real steps followed by 12 learned mask tokens. A latent predictor
estimates representations at all future positions. The EMA target branch sees
the complete unmasked 24-step sequence and supplies stop-gradient targets.

Training, validation, fine-tuning, and testing therefore use the same causal
information boundary. `masked_steps` must equal `pred_steps`, preventing future
ground-truth leakage. The older arbitrary random mask remains available only
through `--pretrain-mask-mode random` as an ablation. Pretraining uses latent L1
and does not directly regress traffic flow.

For cross-city transfer, the default transferred representation is the EMA
target encoder. The online encoder is available as an ablation. The source
sensor embedding and JEPA predictor are never transferred.

## Installation and test

```bash
cd staeformer-jepa
pip install -e . --no-deps
python -m unittest discover -s tests -v
```

The package needs a compatible PyTorch and NumPy. `--no-deps` avoids replacing
the server's CUDA-enabled PyTorch.

## Data

Each city directory must contain:

```text
incident_train.npy
incident_val.npy
incident_test.npy
```

Each object-array sample contains `x_data` and `y_data` with shapes
`[steps, sensors, features]`. Only `--traffic-feature` (default channel 0) is
read. Incident metadata and all other feature channels are ignored. The first
three `y_data` traffic steps are the downstream labels.

## Same-city comparison

```bash
DATA_ROOT=/absolute/path/to/TopoJEPA/dataset/Fremont \
OUTPUT_DIR=runs/fremont_12_to_3 \
bash scripts/run_fremont_12_to_3.sh
```

This compares STAEformer scratch, JEPA frozen transfer, and JEPA full
fine-tuning on Fremont.

## Oakland JEPA -> Fremont STAEformer

Run the complete cross-city experiment:

```bash
SOURCE_ROOT=/absolute/path/to/TopoJEPA/dataset/Oakland \
TARGET_ROOT=/absolute/path/to/TopoJEPA/dataset/Fremont \
OUTPUT_DIR=runs/oakland_to_fremont \
bash scripts/run_oakland_to_fremont.sh
```

For the recommended 12-to-12 causal experiment, run:

```bash
SOURCE_ROOT=/absolute/path/to/TopoJEPA/dataset/Oakland \
TARGET_ROOT=/absolute/path/to/TopoJEPA/dataset/Fremont \
OUTPUT_DIR=runs/oakland_to_fremont_12step \
bash scripts/run_oakland_to_fremont_12step.sh
```

Background execution:

```bash
nohup env \
SOURCE_ROOT=/absolute/path/to/TopoJEPA/dataset/Oakland \
TARGET_ROOT=/absolute/path/to/TopoJEPA/dataset/Fremont \
OUTPUT_DIR=runs/oakland_to_fremont \
bash scripts/run_oakland_to_fremont.sh \
> run_oakland_to_fremont.log 2>&1 &
```

The pipeline performs:

1. Causal future-12-step JEPA pretraining and validation on Oakland.
2. A Fremont STAEformer-from-scratch baseline.
3. EMA-target transfer with shared weights frozen. The fresh Fremont sensor
   embedding and forecast head remain trainable.
4. EMA-target transfer with full fine-tuning. Transferred weights use
   `encoder_lr_scale * finetune_lr`; new target parameters use `finetune_lr`.
5. Final evaluation on Fremont test only, using 12 real historical steps plus
   12 learned mask tokens to predict 12 future flow steps.

To reuse an existing Oakland checkpoint:

```bash
python run_transfer.py \
  --mode finetune \
  --source-data /path/to/Oakland \
  --target-data /path/to/Fremont \
  --pretrained-checkpoint /path/to/best_jepa.pt \
  --transfer-branch target \
  --output runs/oakland_to_fremont
```

Use `--transfer-branch online` to compare the online and EMA target encoders.

## Outputs

```text
runs/oakland_to_fremont/
  source/pretrain/best_jepa.pt
  scratch/best_forecast.pt
  frozen/best_forecast.pt
  full/best_forecast.pt
  transfer_report.json
  all_results.json
  summary.txt
```

Metrics include overall MAE/MSE/RMSE and values for each future horizon. The
stored traffic values remain in their released units unless inverse-scaling
statistics are separately available.

## Attribution

The backbone follows Liu et al., *Spatio-Temporal Adaptive Embedding Makes
Vanilla Transformer SOTA for Traffic Forecasting*, CIKM 2023, with the stated
embedding factorization for cross-city transfer.

```bash
nohup env \
SOURCE_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Oakland \
TARGET_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Fremont \
OUTPUT_DIR=runs/oakland_to_fremont_causal_12step \
bash scripts/run_oakland_to_fremont_12step.sh \
> run_oakland_to_fremont_causal_12step.log 2>&1 &
```

```bash
nohup env \
SOURCE_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Oakland \
TARGET_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Berkeley \
OUTPUT_DIR=runs/oakland_to_berkeley_causal_12step \
bash scripts/run_oakland_to_fremont_12step.sh \
> run_oakland_to_berkeley_causal_12step.log 2>&1 &
```