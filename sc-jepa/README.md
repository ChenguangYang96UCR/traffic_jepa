# SC-JEPA fine-only traffic experiment

This directory adapts the public [SC-JEPA implementation](https://github.com/Echoo113/SC-JEPA)
to Fremont traffic forecasting while removing the modules crossed out in the
experiment diagram. The retained architecture follows upstream commit
`2ea322af6c60c7125d064b3fe157b7bbd611736c`; the traffic window and forecast
head are local adaptations.

## Retained and removed modules

Retained path:

`traffic context patches -> online encoder -> online soft codebook -> fine predictor`

The EMA target path is:

`real future patches -> EMA encoder -> EMA soft codebook -> fine target distribution`

The pretraining objective is only the fine-scale distribution loss:

`L_fine = KL(Pi_target || softmax(fine_logits / prediction_temperature))`

The following crossed modules are absent from both the forward graph and loss:

- reconstruction decoder and reconstruction loss;
- separate codebook-alignment branch and its loss;
- downsample/coarse target, coarse predictor, and coarse KL loss;
- latent prediction head and latent MSE;
- the earlier adaptation's VQ commitment and entropy regularizers.

The online and EMA codebooks visible inside the retained context/target branches
are still required: they construct the fine code distributions compared by KL.
The target encoder and target codebook receive no gradients and are updated from
their online counterparts with EMA 0.996 during pretraining.

## Fremont adaptation

Each sensor is processed independently with shared weights. A `[12, 93]` history
and its real `[12, 93]` future are split into four non-overlapping length-3
patches per sensor. The upstream-style patch MLP and Transformer encoder use
`D=256`, 6 layers, 8 heads and dropout 0.1. The fine predictor uses `H=128`,
2 layers and 4 heads. The soft codebook has 128 entries and temperature 0.1;
the fine KL prediction temperature is 0.8.

This is not masked pretraining. The online branch sees the stored history
`x_data`; the EMA branch sees the paired real future `y_data`. No incident,
topology, timestamp or fabricated future feature is loaded. The dataset filename
remains `incident_{train,val,test}.npy`, but only traffic feature 0 is read.

For downstream prediction, the predicted fine-code probabilities are mapped by
a linear traffic head back to each future length-3 patch, then restored using
history-only mean and standard deviation. Pure SSL leaves this forecast head
untrained, so it has no valid direct-test MAE. A frozen probe trains only this
head; full fine-tuning trains the online encoder, codebook, predictor and head.

## Run

From the repository root:

```bash
conda activate topojepa
bash sc-jepa/run_fremont.sh
```

The default `all` protocol performs fine-KL pretraining, then independently runs
a frozen forecast probe and full fine-tuning from the same pretraining checkpoint.
It uses batch size 32, seed 2026, learning rate 0.0001, and at most 20 epochs per
stage. Checkpoints are selected by validation fine KL during pure pretraining and
validation MSE during probe/full fine-tuning.

```bash
bash sc-jepa/run_fremont.sh --mode pretrain
bash sc-jepa/run_fremont.sh --mode probe
bash sc-jepa/run_fremont.sh --mode full
bash sc-jepa/run_fremont.sh --mode summary
bash sc-jepa/run_fremont.sh --mode eval \
  --checkpoint runs/fremont_fine_only_seed2026/full/checkpoint.pt
```

Existing stage directories are never overwritten. The old SC-inspired
checkpoints are intentionally rejected because their modules and objective differ;
use the new default run directory or another clean `--run-dir` and pretrain again.

### Optional joint forecasting objective

To train the traffic head during pretraining as well, run a separate experiment:

```bash
bash sc-jepa/run_fremont.sh --forecast-weight 1 \
  --run-dir runs/fremont_fine_only_joint_seed2026
```

Its objective is `L_fine + forecast_weight * MSE`. This variant can report a
direct pretraining test result; pure fine-KL SSL cannot.

### Former-test 60/20/20 adaptation protocol

Pretrain on the original train/validation data first. Then use the verified,
deduplicated adaptation split for downstream probe/full fine-tuning:

```bash
bash sc-jepa/run_fremont.sh --mode transfer \
  --checkpoint runs/fremont_fine_only_seed2026/pretrain/checkpoint.pt \
  --adaptation-dir ../TopoJEPA/dataset/Fremont_adaptation_622 \
  --run-dir runs/fremont_fine_only_adaptation_622_seed2026
```

The adaptation manifest is verified before training. Training uses its train
partition, checkpoint selection uses validation, and the held-out test partition
is evaluated only after selection.

## Outputs and evaluation contract

Each stage writes `checkpoint.pt`, `history.json`, and `run.json`; stages with a
trained forecast head also write `metrics.json`. MAE/MSE/RMSE average over all
samples, horizons and sensors in stored traffic units. Fine-tune checkpoints
record the SHA256 of their exact pretraining source. Always compare models using
the same data split, input/target, units, seed protocol and validation criterion.

The previously reported SC values around MAE 0.094 came from the removed
SC-inspired architecture. They are legacy results, not results for this
fine-only implementation; this model must be rerun before updating the table.

## Tests

```bash
cd sc-jepa
python -m unittest discover -s tests -v
```

The tests cover traffic-only loading, shapes, absent crossed modules, frozen EMA
targets, stage freezing, checkpoint round trips, transfer boundaries and the CPU
training/evaluation CLI.
