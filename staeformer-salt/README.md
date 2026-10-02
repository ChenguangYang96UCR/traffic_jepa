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
