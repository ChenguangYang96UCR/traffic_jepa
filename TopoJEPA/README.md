# TopoJEPA

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