# TopoJEPA

For pretraining on original train/val and adapting on a 60/20/20 split of the
former test pool, see [the shared adaptation protocol](../Preprocess/ADAPTATION_622.md).
Both JEPA and SC-JEPA use the same verified split files. Existing behavior stays
unchanged unless the adaptation option is explicitly enabled.

TopoJEPA is a multivariate time-series forecasting. This repository contains the model implementation, data loaders, datasets, precomputed text embeddings, and validation scripts used to reproduce the reported experiments.


## Environment

```bash
conda create -n topojepa python=3.10
conda activate topojepa
pip install -r requirements.txt
```

## Precomputed text targets

The repository already includes the text-embedding caches required by the validation scripts. Each cache stores one normalized 512-dimensional frozen CLIP embedding per window and variable.


To regenerate the caches, run the following commands from the repository root.

### Homeless

```bash
python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/Homeless/ \
  --data_path Homeless.csv \
  --target node_46 \
  --freq w \
  --seq_len 16 \
  --label_len 8 \
  --pred_len 16 \
  --cache_dir ./text_embedding_cache \
  --device cuda
```

### dengue

```bash
python scripts/text_jepa/precompute_clip_embeddings.py \
  --root_path ./dataset/dengue/ \
  --data_path dengue.csv \
  --target VICHADA \
  --freq w \
  --seq_len 16 \
  --label_len 8 \
  --pred_len 16 \
  --cache_dir ./text_embedding_cache \
  --device cuda
```

## Running the experiments

Run commands from the repository root.

```bash
bash scripts/validation/run_Homeless.sh
bash scripts/validation/run_dengue.sh
```

### Fremont traffic-only versus incident-aware JEPA

The traffic-only script ignores all incident fields. The incident-aware script
keeps traffic as the only forecasting target and conditions JEPA on incident
description, type, holiday, relative position, and the three distance features
for each sensor.

```bash
bash scripts/validation/run_Fremont_traffic.sh
bash scripts/validation/run_Fremont_incident.sh
```

Incident conditioning is opt-in through `--incident`. Each Fremont sample must
contain `incident_features`, `incident_position`, and `incident_distances`; the
last field must have shape `[93, 3]`. Description/type cardinalities are read
from `desc_mapping.json` and `type_mapping.json` when available, otherwise they
are inferred from all three data splits.

### Fremont JEPA pre-training and fine-tuning

Gradual unfreezing can reuse an existing joint-pretraining checkpoint:

```bash
bash scripts/validation/run_Fremont_pretrain_finetune.sh gradual
```

Defaults: epochs 1–2 train predictor/head; epochs 3–5 additionally unfreeze
the last encoder block and final norm; epoch 6 onward trains the full online
network. Frozen feature modules stay in eval mode. The pretrained head is kept,
the EMA teacher stays frozen, and only forecasting MSE is optimized. Adam
registers all eventual parameters once, preserving moments across transitions.
Early stopping is enabled only in the full phase, while checkpoint selection
keeps the best validation MSE across all phases. Learning rates and their
schedule are unchanged from the other fine-tuning baselines.

Override durations with GRADUAL_HEAD_EPOCHS and GRADUAL_PARTIAL_EPOCHS;
PARTIAL_LAYERS controls the middle phase. FINETUNE_EPOCHS must exceed the sum
of the first two durations. The all/summary modes include gradual results.

The staged runner trains with forecasting MSE + weighted JEPA loss, directly
tests that model, then initializes full and partial supervised fine-tuning
from the same checkpoint. The online encoder, predictor and trained forecast
head are preserved together; no EMA substitution or head reset is performed.

```bash
bash scripts/validation/run_Fremont_pretrain_finetune.sh all
```

The script prints MAE/MSE/RMSE for direct pretraining evaluation, full and
partial fine-tuning. Default JEPA weight is 0.1 (override with JEPA_WEIGHT).

New run names preserve the old latent-only/probe results. Run pretraining again:
old latent-only checkpoints contain an untrained forecast head. Default budgets
are 20 pretraining and 20 fine-tuning epochs. Compare against uninterrupted
joint training with the same total budget when assessing stage switching.

The stages can also be launched separately:

```bash
bash scripts/validation/run_Fremont_pretrain_finetune.sh pretrain
bash scripts/validation/run_Fremont_pretrain_finetune.sh full
bash scripts/validation/run_Fremont_pretrain_finetune.sh partial
bash scripts/validation/run_Fremont_pretrain_finetune.sh summary
```

During pretraining the online embedding, encoder, predictor and forecast head
receive gradients. The teacher receives EMA updates only. Checkpoint selection
uses validation forecasting MSE. Only-pretrain results are direct predictions
of the saved online network, with no probe training.

Fine-tuning uses forecasting MSE only, without EMA updates.
`full` updates the complete online forecasting path. `partial` freezes the input
embedding and all but the final Transformer layer, while training the final
layer, JEPA predictor, and forecast projector. Set the number of unfrozen final
layers with `PARTIAL_LAYERS`; zero gives a predictor/head-only adaptation.

```bash
PARTIAL_LAYERS=0 FINETUNE_EPOCHS=20 \
  bash scripts/validation/run_Fremont_pretrain_finetune.sh partial
```

The default pretraining checkpoint is
`./checkpoints/Fremont_joint_pretrain_0/checkpoint.pth`. It can be overridden
when fine-tuning:

```bash
PRETRAIN_CHECKPOINT=/path/to/checkpoint.pth \
  bash scripts/validation/run_Fremont_pretrain_finetune.sh full
```

### JEPA-fixed pretraining objective comparison

The objective benchmark keeps the online/EMA encoders, latent predictor,
backbone, data split, optimizer and full fine-tuning protocol fixed. JEPA is
active in every row; only the auxiliary pretraining task changes:

- `jepa`: latent future prediction only;
- `forecast`: clean history forecasting plus JEPA (the previous default);
- `masked_forecast`: forecast from a corrupted history plus JEPA;
- `reconstruction`: reconstruct masked normalized history plus JEPA;
- `forecast_reconstruction`: masked forecasting and reconstruction plus JEPA.

Run all five pretrain-to-full-fine-tune experiments and print one comparison:

```bash
bash scripts/validation/run_Fremont_pretrain_objectives.sh all
```

Run one objective or reprint existing results:

```bash
bash scripts/validation/run_Fremont_pretrain_objectives.sh masked_forecast
bash scripts/validation/run_Fremont_pretrain_objectives.sh summary
```

Masked objectives use a continuous temporal block with ratio `0.25` by
default. Override with `MASK_STRATEGY=random|temporal|sensor|block` and
`MASK_RATIO`. Reconstruction uses a lightweight linear decoder and computes
MSE only at masked positions; that decoder is frozen during fine-tuning.
`jepa` and `reconstruction` do not train the forecast projector, so direct-test
metrics are intentionally reported as `n/a`; their comparable result is the
full fine-tuned metric. Forecast-based checkpoint selection uses ordinary
unweighted validation MSE, while latent/reconstruction-only runs select their
actual validation pretraining loss.

### Learned forecast-loss mask during pretraining

The optional forecast mask learns one context-conditioned weight for every
future time point and sensor, with shape `[batch, pred_len, sensors]`. It is used
only to weight the pretraining forecast MSE; inference outputs are never
multiplied by the mask. A nonzero floor keeps every target trainable, while mean
budget, Bernoulli-entropy, and temporal-smoothness penalties prevent the trivial
all-zero solution.

```bash
bash scripts/validation/run_Fremont_pretrain_finetune_mask.sh all
```

Defaults: floor `0.1`, target mean `0.5`, budget weight `0.1`, entropy weight
`0.01`, and smoothness weight `0.01`. The log reports ordinary validation MSE
(used for checkpoint selection), masked MSE, and mask mean/min/max. Mask runs
use separate tags and do not overwrite the existing TopoJEPA experiments.

### Cross-city pretraining: Alameda cities to Fremont

The cross-city runner uses all non-empty cities in `dataset/Alameda/sensors.csv`
except Fremont for joint traffic+JEPA pretraining. It then evaluates the saved
model zero-shot on Fremont and independently runs full, partial, gradual and
LoRA fine-tuning on Fremont train/validation/test splits:

```bash
bash scripts/validation/run_AlamedaCities_pretrain_Fremont_finetune.sh all
```

No Fremont sensor is present in the default pretraining input. The released
Alameda history/future sample pairs and its original train/validation/test
boundaries are preserved; only the sensor axis is filtered. Incident fields,
topology and time covariates remain disabled. Because the inverted architecture
shares its encoder/projector across variable tokens, its learned parameters do
not depend on the number of selected sensors and load strictly into the
93-sensor Fremont model.

To pretrain on a named subset rather than every other city:

```bash
ALAMEDA_CITIES='Oakland,Hayward' \
  bash scripts/validation/run_AlamedaCities_pretrain_Fremont_finetune.sh all
```

Available modes are `pretrain`, `zero-shot`, `full`, `partial`, `gradual`,
`lora`, `summary`, and `all`. `PRETRAIN_CHECKPOINT` can select an existing
cross-city checkpoint. The summary contains only metrics evaluated on the
Fremont test set; the pretraining run's Alameda test metric is not mixed into
the cross-city comparison table.

### Per-city same-domain pretraining and fine-tuning

To run a separate pretrain → full-fine-tune experiment for every non-empty City
in Alameda `sensors.csv`:

```bash
bash scripts/validation/run_each_AlamedaCity_pretrain_finetune.sh all
```

For each city, both stages use only that city's sensor columns. Pretraining uses
that city's Alameda train/validation splits and directly evaluates its test
split. Fine-tuning starts from the matching city checkpoint, reuses that city's
train/validation data with forecasting MSE, and evaluates the same city test
split. Thus this is a same-domain staged-training experiment, not a held-out-city
transfer experiment.

Limit a run to named cities or select a different fine-tuning strategy:

```bash
CITIES='Fremont,Oakland,Hayward' \
  bash scripts/validation/run_each_AlamedaCity_pretrain_finetune.sh all

FINETUNE_STRATEGY=partial \
  bash scripts/validation/run_each_AlamedaCity_pretrain_finetune.sh all
```

`FINETUNE_STRATEGY` supports `full` (default), `partial`, `gradual`, and `lora`.
Use modes `pretrain` and `finetune` to separate stages, or `summary` to print two
rows per city: direct joint-pretraining performance and fine-tuned performance.
The script passes each city's real sensor count to the logged `enc_in/dec_in/c_out`
fields, although the inverted model itself supports a runtime-variable token count.

## LoRA fine-tuning

From `TopoJEPA`, reuse the existing joint-pretrained checkpoint without rerunning
pretraining:

```bash
bash scripts/validation/run_Fremont_pretrain_finetune.sh lora
# Alternative rank (also use these variables when requesting summary later):
LORA_RANK=4 LORA_ALPHA=8 bash scripts/validation/run_Fremont_pretrain_finetune.sh lora
```

Default: rank 8, alpha 16, adapter dropout 0, adapter/head learning rate 0.0001,
20 maximum epochs, validation early stopping. Override `PRETRAIN_CHECKPOINT`
for a different base checkpoint. `all` includes LoRA and `summary` includes its
MAE/MSE/RMSE. Each rank/alpha/dropout/LR-scale combination has its own result tag.

The online encoder's attention Q/V projections use `W(x) + (alpha/r) B A x`.
Only A/B and the existing forecast projector (plus incident output head when
enabled) are trained. Embedding, original encoder weights, latent predictor,
incident/graph feature modules and EMA teacher remain frozen. Fine-tuning uses
forecast MSE only, with no EMA updates. B starts at zero, preserving the loaded
forecast exactly before adaptation. Backbone dropout is disabled; adapter
dropout is separately configurable. No PEFT dependency is required.

LoRA checkpoints contain the frozen base and adapters together, not adapter-only
weights. For standalone evaluation use `--is_training 0 --training_stage finetune
--finetune_strategy lora`, matching architecture, rank, and experiment tag;
the wrapped architecture is rebuilt before strict checkpoint loading. The base
pretraining checkpoint is needed to start training, not standalone evaluation.
LoRA reduces trainable parameters; it does not guarantee better accuracy or a
proportional reduction in activation memory. Compare using the same data,
pretrained checkpoint, validation protocol and multiple random seeds.
