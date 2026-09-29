#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Full SC-JEPA method and Appendix-B optimizer/loss settings, adapted to the
# available Fremont 12->12 windows (4 patches of length 3). This is a traffic
# transfer experiment, not a reproduction of the paper's anomaly classifier.
python -u train.py \
  --architecture scjepa_paper_full_v1 \
  --data-dir ../TopoJEPA/dataset/Fremont \
  --run-dir runs/fremont_paper_method_seed2026 \
  --seq-len 12 --pred-len 12 --patch-len 3 \
  --dim 256 --codes 128 \
  --encoder-hidden 64 --encoder-bottleneck 32 \
  --heads 8 --layers 6 --dropout 0.1 \
  --predictor-dim 128 --predictor-heads 4 --predictor-layers 2 \
  --code-temperature 0.1 --prediction-temperature 0.8 --ema 0.996 \
  --batch-size 128 --lr 0.0005 --weight-decay 0.00001 --lr-gamma 1 \
  --grad-clip 0.5 --pretrain-epochs 100 --selection-start-epoch 50 --patience 10 \
  --fine-weight 1 --coarse-weight 0.5 --latent-weight 0.1 \
  --embedding-weight 1 --commitment-weight 0.25 \
  --sample-entropy-weight 0.005 --batch-entropy-weight 0.01 \
  --recon-weight-start 0.5 --recon-weight-end 0.1 \
  --forecast-weight 0 --paper-train-split --channel-independent-loader \
  "$@"
