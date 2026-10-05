# Three controlled STAEformer-SALT experiments

The package evaluates three distinct sources of benefit:

```text
Experiment 1: Fremont Teacher -> Fremont Student -> Fremont forecasting
Experiment 2: Oakland Teacher -> Fremont Student -> Fremont forecasting
Experiment 3: Oakland Teacher -> Oakland Student -> Fremont transfer/fine-tuning
```

Experiment 1 is fully in-domain. Comparing Experiments 1 and 2 isolates the
effect of the Teacher's training city because the Student and downstream data
remain Fremont. Comparing Experiments 2 and 3 isolates the effect of the
Student's training city because both use the exact same frozen Oakland Teacher.

All Teachers use the same node-agnostic architecture
(`teacher_sensor_dim=0`). This is necessary because the frozen Oakland Teacher
in Experiment 2 must process Fremont inputs with a different node count.
Students retain city-specific sensor embeddings.

Each experiment uses only traffic channel 0 and predicts future 12 steps from
past 12 steps. Incident, time-of-day, and day-of-week features are not used.

## Run all experiments

```bash
FREMONT_ROOT=/absolute/path/to/Fremont \
OAKLAND_ROOT=/absolute/path/to/Oakland \
RUN_ROOT=runs/three_city_experiments \
bash scripts/run_three_experiments.sh
```

For a background run:

```bash
nohup env \
FREMONT_ROOT=/absolute/path/to/Fremont \
OAKLAND_ROOT=/absolute/path/to/Oakland \
RUN_ROOT=runs/three_city_experiments \
bash scripts/run_three_experiments.sh \
> run_three_experiments.log 2>&1 &
```

The Oakland Teacher is trained once and its checkpoint is shared by
Experiments 2 and 3. Experiment 1 trains a separate Fremont Teacher with the
same architecture and hyperparameters.

## Individual commands

Experiment 1:

```bash
FREMONT_ROOT=/path/to/Fremont \
bash scripts/run_fremont_teacher_fremont_student.sh
```

Train the shared Oakland Teacher:

```bash
OAKLAND_ROOT=/path/to/Oakland \
bash scripts/train_oakland_teacher.sh
```

Experiment 2:

```bash
TEACHER_CHECKPOINT=runs/oakland_teacher/teacher/best_teacher.pt \
FREMONT_ROOT=/path/to/Fremont \
bash scripts/run_oakland_teacher_fremont_student.sh
```

Experiment 3:

```bash
TEACHER_CHECKPOINT=runs/oakland_teacher/teacher/best_teacher.pt \
SOURCE_ROOT=/path/to/Oakland \
TARGET_ROOT=/path/to/Fremont \
bash scripts/run_oakland_to_fremont.sh
```

## Evaluation

Every experiment reports:

```text
STAEformer scratch
SALT Student frozen probe
SALT Student full fine-tuning
```

The combined report is written to:

```text
runs/three_city_experiments/three_experiment_summary.txt
```

## Three Berkeley experiments

Berkeley is an independently extracted dataset. The Berkeley experiment suite
mirrors the three Fremont experiments:

```text
1. Berkeley Teacher -> Berkeley Student -> Berkeley forecasting
2. Oakland Teacher  -> Berkeley Student -> Berkeley forecasting
3. Oakland Teacher  -> Oakland Student  -> Berkeley transfer/fine-tuning
```

Experiments 2 and 3 reuse the same already-trained, frozen Oakland Teacher.
They do not retrain it. Each experiment reports scratch, frozen, and full
fine-tuning performance on the Berkeley test split.

Expected independent Berkeley files:

```text
Berkeley/
  incident_train.npy
  incident_val.npy
  incident_test.npy
  sensors.csv
```

Run it with:

```bash
nohup env \
BERKELEY_ROOT=/absolute/path/to/Berkeley \
OAKLAND_ROOT=/absolute/path/to/Oakland \
OAKLAND_TEACHER_CHECKPOINT=/absolute/path/to/best_teacher.pt \
RUN_ROOT=runs/three_berkeley_experiments \
bash scripts/run_three_berkeley_experiments.sh \
> run_three_berkeley_experiments.log 2>&1 &
```

The combined table is written to
`runs/three_berkeley_experiments/three_experiment_summary.txt` by default.

## Future-focused SALT ablation

This experiment changes both stages in a controlled way.

### Stage 1: future-biased Teacher reconstruction

The Teacher still receives the complete 24-step sequence (12 history + 12
future), but its spatio-temporal reconstruction blocks are allocated by region.
The default controlled setting uses four blocks:

```text
3 blocks entirely inside future (75%)
1 block entirely inside history (25%)
```

This prevents a block from straddling the history/future boundary and makes the
Teacher reconstruction objective explicitly future-focused. Training logs and
`teacher_metrics.json` report history loss, future loss, and the realized
future-mask fraction.

### Stage 2: two Student latent objectives

Both Students see the same input: observed history plus 12 masked future
positions. The frozen Teacher sees the complete sequence.

```text
all:    L = MAE(P(Student)[history + future], Teacher[history + future])
future: L = MAE(P(Student)[future],           Teacher[future])
```

`future` is the behavior of the earlier implementation. `all` is the new
comparison. The validation log always reports `val_all`, `val_history`, and
`val_future`, even when only one scope is optimized.

The full Berkeley ablation runs both Student losses for all three city setups:

```bash
nohup env \
BERKELEY_ROOT=/absolute/path/to/Berkeley \
OAKLAND_ROOT=/absolute/path/to/Oakland \
RUN_ROOT=runs/future_focus_berkeley \
bash scripts/run_future_focus_berkeley_ablation.sh \
> run_future_focus_berkeley.log 2>&1 &
```

Because the Teacher objective has changed, the script trains new
future-focused Berkeley and Oakland Teachers once. A legacy uniform-mask
Oakland checkpoint is rejected by the controlled Student runs. To reuse an
already-trained future-focused checkpoint, set `OAKLAND_TEACHER_CHECKPOINT`.

The consolidated result is written to:

```text
runs/future_focus_berkeley/future_focus_summary.txt
```

## Supervised STAEformer transfer baseline

This baseline separates the effect of SALT from ordinary supervised source-city
pretraining:

```text
Oakland history -> STAEformer -> Oakland future flow
                         |
                         +-- transfer shared encoder weights
                                      |
                                      v
Berkeley history -> transferred encoder -> new Berkeley forecast head
```

The Oakland sensor embedding and forecast head are not transferred. Berkeley
receives a newly initialized sensor embedding and forecasting head. Shared
input, step, temporal-attention, spatial-attention, normalization, and FFN
weights are transferred. Evaluation reports:

```text
Berkeley STAEformer scratch
Oakland supervised encoder -> Berkeley frozen
Oakland supervised encoder -> Berkeley full fine-tuning
```

Run it with the same architecture and training budget as SALT:

```bash
nohup env \
OAKLAND_ROOT=/absolute/path/to/Oakland \
BERKELEY_ROOT=/absolute/path/to/Berkeley \
OUTPUT_DIR=runs/oakland_supervised_to_berkeley \
SALT_ROOT=runs/future_focus_berkeley \
bash scripts/run_oakland_supervised_to_berkeley.sh \
> run_oakland_supervised_to_berkeley.log 2>&1 &
```

The supervised baseline is written to:

```text
runs/oakland_supervised_to_berkeley/summary.txt
```

If the future-focused SALT results exist under `SALT_ROOT`, the script also
writes the direct comparison to:

```text
runs/oakland_supervised_to_berkeley/supervised_vs_salt.txt
```

## Oakland Teacher width ablation: 32 vs 128

This focused experiment reproduces the strongest future-focused configuration:

```text
Oakland Teacher -> Berkeley Student
Student latent loss = all steps
Downstream = Berkeley frozen and full fine-tuning
```

Only the Teacher capacity changes. The original node-agnostic Teacher has model
dimension 32 (`16 traffic + 16 step`). The wider Teacher has dimension 128:

```text
64 traffic embedding + 64 step embedding + 0 sensor embedding
3 Transformer layers, 4 heads, FFN dimension 256
```

The Student and downstream STAEformer remain unchanged at dimension 128. Run:

```bash
nohup env \
OAKLAND_ROOT=/absolute/path/to/Oakland \
BERKELEY_ROOT=/absolute/path/to/Berkeley \
OUTPUT_DIR=runs/oakland_teacher128_berkeley_all \
BASELINE_SALT_ROOT=runs/future_focus_berkeley \
bash scripts/run_oakland_teacher128_berkeley_all.sh \
> run_oakland_teacher128_berkeley_all.log 2>&1 &
```

The new result and direct d=32/d=128 comparison are written to:

```text
runs/oakland_teacher128_berkeley_all/summary.txt
runs/oakland_teacher128_berkeley_all/teacher_width_comparison.txt
```

## Oakland Teacher -> Berkeley Student hyperparameter sweep

This sweep keeps the controlled setting fixed at a node-agnostic 128-d Oakland
Teacher and an all-step Berkeley Student loss. It varies:

```text
Teacher mask blocks:        4, 8
Teacher future-block ratio: 0.50, 0.75, 1.00
Berkeley full-FT LR:        1e-4, 5e-4, 1e-3
```

Each `(mask blocks, future ratio)` pair trains one Teacher and one Student;
those checkpoints are reused for all three fine-tuning rates. The scratch
baseline is also trained only once. The best configuration is selected by
Berkeley validation MAE, never by test MAE.

```bash
nohup env \
OAKLAND_ROOT=/absolute/path/to/Oakland \
BERKELEY_ROOT=/absolute/path/to/Berkeley \
RUN_ROOT=runs/oakland_teacher128_berkeley_sweep \
MASK_BLOCKS_LIST="4 8" \
FUTURE_RATIOS="0.5 0.75 1.0" \
FINETUNE_LRS="0.0001 0.0005 0.001" \
bash scripts/run_oakland_teacher128_berkeley_sweep.sh \
> run_oakland_teacher128_berkeley_sweep.log 2>&1 &
```

The script is restartable and skips completed checkpoints/results. It writes:

```text
runs/oakland_teacher128_berkeley_sweep/sweep_summary.txt
runs/oakland_teacher128_berkeley_sweep/best_config.json
```

## Best-SALT objective and checkpoint-selection ablation

This pipeline fixes the best sweep configuration:

```text
Oakland Teacher dimension = 128
Teacher mask blocks       = 8
Teacher future ratio      = 0.50
Berkeley Student scope    = all steps
Downstream full-FT LR     = 1e-3
```

It compares three Student methods:

```text
A  latent L1 training; select the Student checkpoint by validation latent L1
B  same A training trajectory; select by an online detached probe's validation MAE
C  forecast MAE + 0.1 * latent L1; select by validation forecast MAE
```

A and B share one Student training run and differ only in checkpoint selection.
The B probe receives detached Student features, so it cannot update the encoder.
C uses future mask tokens and never reads future target values in its Student
branch. Its forecasting head warm-starts the common downstream full fine-tune;
set `--c-head-init fresh` in `run_abc_ablation.py` for an encoder-only control.

Run:

```bash
nohup env \
BERKELEY_ROOT=/absolute/path/to/Berkeley \
SWEEP_ROOT=runs/oakland_teacher128_berkeley_sweep \
OUTPUT_DIR=runs/best_salt_abc \
LATENT_WEIGHT=0.1 \
bash scripts/run_best_salt_abc.sh \
> run_best_salt_abc.log 2>&1 &
```

The output includes the reported best SALT row supplied for comparison:

```text
runs/best_salt_abc/abc_summary.txt
runs/best_salt_abc/abc_results.json
```


```bash
nohup env \
BERKELEY_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Berkeley \
OUTPUT_DIR=runs/best_salt_abc \
bash scripts/run_best_salt_abc.sh \
> run_best_salt_abc.log 2>&1 &
```

```bash
python run_abc_ablation.py \
  --teacher-checkpoint /home/ADS/cyang314/ucr_work/traffic_jepa/staeformer-salt-future-focus/runs/oakland_teacher128_berkeley_sweep/teachers/blocks=8_ratio=0.5/teacher/best_teacher.pt \
  --berkeley-data /home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Berkeley \
  --output runs/best_salt_abc \
  --student-epochs 100 \
  --finetune-epochs 50 \
  --finetune-lr 0.001 \
  --c-head-init fresh
```

```bash
nohup env \
BERKELEY_ROOT=/home/ADS/cyang314/ucr_work/traffic_jepa/traffic_forcasting/TopoJEPA/dataset/Berkeley \
SWEEP_ROOT=runs/oakland_teacher128_berkeley_sweep \
OUTPUT_ROOT=runs/fresh_head_latent_weight_sweep \
LATENT_WEIGHTS="0 0.01 0.03 0.1 0.3 1.0 3.0" \
bash scripts/run_fresh_head_latent_weight_sweep.sh \
> run_fresh_head_latent_weight_sweep.log 2>&1 &
```
