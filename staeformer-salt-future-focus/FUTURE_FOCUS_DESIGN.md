# Future-focused SALT for traffic forecasting

## Research question

Traffic representation learning should not require history and future tokens to
contribute equally. The downstream task predicts future flow, so this design
tests whether shifting Teacher capacity and Student supervision toward future
positions improves forecasting.

## Stage 1: future-biased Teacher

For a sample with history `H` and future `F`, construct:

```text
X = concat(H, F), shape [batch, 24, sensors, channels]
```

The default mask has four spatio-temporal blocks. Three are sampled completely
inside the future region and one completely inside history. No block crosses
the history/future boundary.

```text
B = 4, future_block_ratio = 0.75
B_future = 3, B_history = 1
```

The Teacher encoder and reconstruction decoder are optimized only at masked
locations:

```text
Z_T_masked = E_T(mask(X))
X_hat      = D_T(Z_T_masked)
L_teacher  = SmoothL1(X_hat[M], X[M])
```

The actual fraction of masked values in future can differ slightly from 0.75
because randomly sized blocks may overlap. It is measured and written to
`teacher_metrics.json`.

## Stage 2: frozen Teacher and two Student objectives

The trained Teacher is frozen. It sees the complete clean sequence. The Student
sees history plus learned mask tokens at all future positions:

```text
Z_T = E_T(concat(H, F))
Z_S = E_S(concat(H, MASK_future))
P_S = Predictor(Z_S)
```

Two independent Students are trained from the same initialization seed.

### A. All-step latent matching

```text
L_all = MAE(P_S[:, 0:24], Z_T[:, 0:24])
```

This gives history and future equal token-level weight. Because the Teacher uses
bidirectional temporal attention, its history latents may contain information
from true future values that the Student cannot observe. This is intentional in
the ablation: it tests whether matching the complete Teacher representation is
helpful or creates an irreducible target.

### B. Future-only latent matching

```text
L_future = MAE(P_S[:, 12:24], Z_T[:, 12:24])
```

This is more directly aligned with forecasting and matches the behavior of the
previous implementation. Training logs report all/history/future validation
MAE for both methods, so the optimized loss and non-optimized regions remain
visible.

## Downstream evaluation

The predictor used only for distillation is discarded. The Student encoder is
attached to the unchanged masked-future STAEformer forecasting head. Evaluation
contains:

```text
STAEformer scratch (trained once on Berkeley)
SALT Student frozen + forecasting head
SALT Student full fine-tuning
```

Both Student objectives are tested under three city configurations:

```text
1. Berkeley Teacher -> Berkeley Student -> Berkeley
2. Oakland Teacher  -> Berkeley Student -> Berkeley
3. Oakland Teacher  -> Oakland Student  -> Berkeley transfer
```

This produces 12 SALT rows (`3 configurations x 2 latent losses x 2 downstream
strategies`) plus one common Berkeley scratch baseline.

## Controlled variables

- Same input/prediction lengths: 12 -> 12.
- Same model widths, optimizer, batch size, epoch budget, and seed.
- Same Berkeley train/validation/test split for downstream evaluation.
- Each future-focused Teacher is trained once and shared by the two Student
  objectives.
- Scratch is trained once because the downstream dataset and seed are shared.
- Legacy uniform-mask Teacher checkpoints are rejected in controlled runs.

## Primary comparison

Within each city configuration, compare `all` and `future` using:

- Student validation latent loss, with care because the objectives average over
  different numbers of tokens.
- Frozen downstream MAE/MSE/RMSE, which most directly measures representation
  quality.
- Full fine-tuning MAE/MSE/RMSE, which measures initialization quality.
- Per-horizon downstream metrics in each `test_metrics.json`.

Do not directly conclude that the method with the smaller raw latent loss is
better: `L_all` and `L_future` have different target sets. Downstream Berkeley
forecasting metrics are the decisive comparison.
