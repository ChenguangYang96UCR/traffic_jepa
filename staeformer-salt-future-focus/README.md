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
