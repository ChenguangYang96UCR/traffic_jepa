#!/usr/bin/env python3
"""Compare three Student objectives on the best Oakland-to-Berkeley SALT setup.

A: latent L1 training, checkpoint selected by validation latent L1.
B: identical latent L1 training, checkpoint selected by an online detached
   forecasting probe on validation MAE.
C: joint forecasting + latent L1 training, selected by validation forecast MAE.

All methods receive the same downstream full fine-tuning procedure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from experiment import (
    datasets_for,
    inspect_dataset,
    make_encoder,
    make_loaders,
    train_downstream,
)
from run import inspect_teacher_checkpoint
from staeformer_salt.distillation import SALTDistiller
from staeformer_salt.model import STAEformerEncoder
from staeformer_salt.training import MetricAccumulator, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-SALT A/B/C ablation")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--berkeley-data", required=True)
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--pred-steps", type=int, default=12)

    # Student STAEformer, fixed to the architecture used by the best SALT run.
    parser.add_argument("--input-embedding-dim", type=int, default=24)
    parser.add_argument("--step-embedding-dim", type=int, default=24)
    parser.add_argument("--sensor-embedding-dim", type=int, default=80)
    parser.add_argument("--feed-forward-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--student-epochs", type=int, default=50)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument(
        "--latent-weight",
        type=float,
        default=0.1,
        help="C objective: forecast MAE + latent_weight * latent L1",
    )
    parser.add_argument(
        "--c-head-init",
        choices=("warm", "fresh"),
        default="warm",
        help="Warm-start C downstream with its task-aware forecasting head",
    )
    parser.add_argument(
        "--only-c",
        action="store_true",
        help="Run only the task-aware C method (used by latent-weight sweeps)",
    )
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--forecast-loss", choices=("mae", "mse"), default="mae")
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/best_salt_abc")

    # Reported best row supplied by the user. This is report-only.
    parser.add_argument("--reference-mae", type=float, default=0.0540987)
    parser.add_argument("--reference-mse", type=float, default=0.0122153)
    parser.add_argument("--reference-rmse", type=float, default=0.1105226)
    parser.add_argument("--reference-val-mae", type=float, default=0.0537047)
    parser.add_argument("--reference-val-latent", type=float, default=0.0215294)
    return parser.parse_args()


def make_teacher_distiller(
    args: argparse.Namespace,
    teacher_checkpoint: Path,
    teacher_nodes: int,
    student_nodes: int,
    channels: int,
    device: torch.device,
) -> SALTDistiller:
    saved = torch.load(teacher_checkpoint, map_location="cpu")
    saved_args = saved.get("args", {})
    encoder_state = saved["encoder"]
    max_steps = int(encoder_state["relative_step_embedding"].shape[0])
    teacher = STAEformerEncoder(
        num_nodes=teacher_nodes,
        max_steps=max_steps,
        input_dim=channels,
        input_embedding_dim=saved_args["teacher_input_dim"],
        step_embedding_dim=saved_args["teacher_step_dim"],
        sensor_embedding_dim=saved_args.get("teacher_sensor_dim", 0),
        feed_forward_dim=saved_args["teacher_ff_dim"],
        num_heads=saved_args["teacher_heads"],
        num_layers=saved_args["teacher_layers"],
        dropout=saved_args.get("dropout", args.dropout),
    )
    teacher.load_state_dict(encoder_state)
    student = make_encoder(args, student_nodes, channels)
    return SALTDistiller(
        teacher,
        student,
        args.input_steps,
        args.pred_steps,
        args.dropout,
        distill_scope="all",
    ).to(device)


def forecast_loss(prediction: torch.Tensor, target: torch.Tensor, name: str):
    if name == "mse":
        return torch.nn.functional.mse_loss(prediction, target)
    return torch.nn.functional.l1_loss(prediction, target)


def make_forecast_head(model, channels, device, seed):
    """Initialize a head without perturbing Student dropout or loader RNG."""
    cuda_devices = (
        []
        if device.type != "cuda"
        else [device.index if device.index is not None else torch.cuda.current_device()]
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed + 20_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 20_000)
        return nn.Linear(model.student.model_dim, channels).to(device)


@torch.no_grad()
def evaluate_student_and_head(model, head, loader, device, pred_steps):
    model.eval()
    head.eval()
    latent_sum = 0.0
    latent_count = 0
    metrics = MetricAccumulator(pred_steps)
    for history, future, labels in loader:
        history = history.to(device)
        future = future.to(device)
        labels = labels.to(device)
        predicted, target, hidden = model.latent_triplet(history, future)
        predicted, target = model.select_scope(predicted, target)
        latent_sum += float((predicted - target).abs().sum())
        latent_count += target.numel()
        metrics.update(head(hidden[:, model.input_steps :]), labels)
    return {
        "latent_l1": latent_sum / latent_count,
        "forecast": metrics.result(),
    }


def checkpoint_payload(model, head, epoch, validation, selection, args):
    return {
        "student_encoder": model.student.state_dict(),
        "predictor": model.predictor.state_dict(),
        "forecast_head": head.state_dict(),
        "epoch": epoch,
        "validation": validation,
        "selection": selection,
        "args": vars(args),
    }


def train_a_b(args, loaders, teacher_checkpoint, teacher_nodes, nodes, channels, device, output):
    """Train once, then select A and B from the same epoch trajectory."""
    a_path = output / "students" / "A_latent_selection.pt"
    b_path = output / "students" / "B_probe_selection.pt"
    if a_path.exists() and b_path.exists():
        print(f"Reusing {a_path} and {b_path}")
        return a_path, b_path

    seed_everything(args.seed + 1)
    model = make_teacher_distiller(
        args, teacher_checkpoint, teacher_nodes, nodes, channels, device
    )
    probe = make_forecast_head(model, channels, device, args.seed)
    student_optimizer = torch.optim.AdamW(
        list(model.student.parameters()) + list(model.predictor.parameters()),
        lr=args.student_lr,
        weight_decay=args.weight_decay,
    )
    probe_optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_lr, weight_decay=args.weight_decay
    )
    a_path.parent.mkdir(parents=True, exist_ok=True)
    best_latent = best_probe = math.inf
    stale_latent = stale_probe = 0
    history_log = []

    for epoch in range(1, args.student_epochs + 1):
        model.train()
        model.teacher.eval()
        latent_total = probe_total = 0.0
        latent_count = probe_count = 0
        for history, future, labels in loaders["train"]:
            history = history.to(device)
            future = future.to(device)
            labels = labels.to(device)

            predicted, target, _ = model.latent_triplet(history, future)
            predicted_scope, target_scope = model.select_scope(predicted, target)
            latent = torch.nn.functional.l1_loss(predicted_scope, target_scope)
            student_optimizer.zero_grad(set_to_none=True)
            latent.backward()
            parameters = list(model.student.parameters()) + list(model.predictor.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            student_optimizer.step()
            latent_total += float(latent.detach()) * target_scope.numel()
            latent_count += target_scope.numel()

            # The online probe tracks forecast utility but cannot update Student.
            model.student.eval()
            with torch.no_grad():
                hidden = model.student_hidden(history).detach()
            model.student.train()
            prediction = probe(hidden[:, args.input_steps :])
            probe_loss = forecast_loss(prediction, labels, args.forecast_loss)
            probe_optimizer.zero_grad(set_to_none=True)
            probe_loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
            probe_optimizer.step()
            probe_total += float(probe_loss.detach()) * labels.numel()
            probe_count += labels.numel()

        validation = evaluate_student_and_head(
            model, probe, loaders["val"], device, args.pred_steps
        )
        history_log.append({"epoch": epoch, **validation})
        print(
            f"A/B {epoch:03d}/{args.student_epochs}: "
            f"train_latent={latent_total / latent_count:.6f} "
            f"train_probe={probe_total / probe_count:.6f} "
            f"val_latent={validation['latent_l1']:.6f} "
            f"val_probe_mae={validation['forecast']['mae']:.6f}"
        )
        if validation["latent_l1"] < best_latent:
            best_latent = validation["latent_l1"]
            stale_latent = 0
            torch.save(
                checkpoint_payload(
                    model, probe, epoch, validation, "validation latent L1", args
                ),
                a_path,
            )
        else:
            stale_latent += 1
        if validation["forecast"]["mae"] < best_probe:
            best_probe = validation["forecast"]["mae"]
            stale_probe = 0
            torch.save(
                checkpoint_payload(
                    model, probe, epoch, validation, "validation probe MAE", args
                ),
                b_path,
            )
        else:
            stale_probe += 1
        if stale_latent >= args.patience and stale_probe >= args.patience:
            print(f"Early stopping A/B after epoch {epoch}")
            break

    write_json(output / "students" / "A_B_training_history.json", history_log)
    return a_path, b_path


def train_c(args, loaders, teacher_checkpoint, teacher_nodes, nodes, channels, device, output):
    checkpoint = output / "students" / "C_task_aware.pt"
    if checkpoint.exists():
        print(f"Reusing {checkpoint}")
        return checkpoint

    seed_everything(args.seed + 1)
    model = make_teacher_distiller(
        args, teacher_checkpoint, teacher_nodes, nodes, channels, device
    )
    head = make_forecast_head(model, channels, device, args.seed)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.student.parameters()) + list(model.predictor.parameters()),
                "lr": args.student_lr,
            },
            {"params": head.parameters(), "lr": args.probe_lr},
        ],
        weight_decay=args.weight_decay,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, stale = math.inf, 0
    history_log = []

    for epoch in range(1, args.student_epochs + 1):
        model.train()
        model.teacher.eval()
        head.train()
        total = latent_total = forecast_total = 0.0
        sample_count = 0
        for history, future, labels in loaders["train"]:
            history = history.to(device)
            future = future.to(device)
            labels = labels.to(device)
            predicted, target, hidden = model.latent_triplet(history, future)
            predicted_scope, target_scope = model.select_scope(predicted, target)
            latent = torch.nn.functional.l1_loss(predicted_scope, target_scope)
            forecast = forecast_loss(
                head(hidden[:, args.input_steps :]), labels, args.forecast_loss
            )
            loss = forecast + args.latent_weight * latent
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            batch = labels.shape[0]
            total += float(loss.detach()) * batch
            latent_total += float(latent.detach()) * batch
            forecast_total += float(forecast.detach()) * batch
            sample_count += batch

        validation = evaluate_student_and_head(
            model, head, loaders["val"], device, args.pred_steps
        )
        history_log.append({"epoch": epoch, **validation})
        print(
            f"C {epoch:03d}/{args.student_epochs}: "
            f"train_total={total / sample_count:.6f} "
            f"train_latent={latent_total / sample_count:.6f} "
            f"train_forecast={forecast_total / sample_count:.6f} "
            f"val_latent={validation['latent_l1']:.6f} "
            f"val_forecast_mae={validation['forecast']['mae']:.6f}"
        )
        if validation["forecast"]["mae"] < best:
            best = validation["forecast"]["mae"]
            stale = 0
            torch.save(
                checkpoint_payload(
                    model,
                    head,
                    epoch,
                    validation,
                    "validation task-aware forecast MAE",
                    args,
                ),
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping C after epoch {epoch}")
                break
    write_json(output / "students" / "C_training_history.json", history_log)
    return checkpoint


def downstream_result(args, checkpoint, loaders, nodes, channels, device, output, name):
    run_name = f"C_{args.c_head_init}" if name == "C" else name
    metrics_path = output / "downstream" / run_name / "full" / "test_metrics.json"
    if metrics_path.exists():
        print(f"Reusing {metrics_path}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = torch.load(checkpoint, map_location="cpu")
    state = saved["student_encoder"]
    encoder = make_encoder(args, nodes, channels)
    encoder.load_state_dict(state)
    initial_head = (
        saved["forecast_head"]
        if name == "C" and args.c_head_init == "warm"
        else None
    )
    return train_downstream(
        args,
        loaders,
        encoder,
        "full",
        device,
        output / "downstream" / run_name,
        initial_head_state=initial_head,
    )


def summarize(args, output, checkpoints, results):
    rows = []
    for name, label in (
        ("A", "A: latent selection"),
        ("B", "B: forecast-probe selection"),
        ("C", f"C: task-aware ({args.c_head_init})"),
    ):
        if name not in checkpoints:
            continue
        state = torch.load(checkpoints[name], map_location="cpu")
        validation = state["validation"]
        result = results[name]
        rows.append({
            "method": label,
            "student_epoch": state["epoch"],
            "student_val_latent": validation["latent_l1"],
            "student_val_forecast_mae": validation["forecast"]["mae"],
            "finetune_val_mae": result["selection_validation"]["mae"],
            "test_mae": result["mae"],
            "test_mse": result["mse"],
            "test_rmse": result["rmse"],
        })
    reference = {
        "method": "Reported best SALT",
        "student_epoch": None,
        "student_val_latent": args.reference_val_latent,
        "student_val_forecast_mae": None,
        "finetune_val_mae": args.reference_val_mae,
        "test_mae": args.reference_mae,
        "test_mse": args.reference_mse,
        "test_rmse": args.reference_rmse,
    }
    all_rows = [reference, *rows]
    lines = [
        "Best SALT objective and checkpoint-selection comparison",
        "Fixed: Teacher d=128, blocks=8, future ratio=0.50, all-step, full FT LR=1e-3",
        "",
        (
            f"{'Method':<31} {'Epoch':>6} {'Stu val latent':>15} "
            f"{'Stu val fcst':>13} {'FT val MAE':>11} {'Test MAE':>11} "
            f"{'Test MSE':>11} {'Test RMSE':>11}"
        ),
        "-" * 120,
    ]
    for row in all_rows:
        def value(key, digits=7):
            item = row[key]
            return "n/a" if item is None else f"{item:.{digits}f}"

        epoch = "n/a" if row["student_epoch"] is None else str(row["student_epoch"])
        lines.append(
            f"{row['method']:<31} {epoch:>6} "
            f"{value('student_val_latent'):>15} "
            f"{value('student_val_forecast_mae'):>13} "
            f"{value('finetune_val_mae'):>11} "
            f"{value('test_mae'):>11} {value('test_mse'):>11} "
            f"{value('test_rmse'):>11}"
        )
    summary = "\n".join(lines)
    print("\n" + summary)
    (output / "abc_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_json(output / "abc_results.json", {
        "fixed_configuration": {
            "teacher_dim": 128,
            "mask_blocks": 8,
            "teacher_future_block_ratio": 0.5,
            "student_distill_scope": "all",
            "finetune_lr": args.finetune_lr,
            "C_latent_weight": args.latent_weight,
            "C_downstream_head_init": args.c_head_init,
        },
        "rows": all_rows,
    })


def main() -> None:
    args = parse_args()
    if args.latent_weight < 0:
        raise ValueError("--latent-weight must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    teacher_checkpoint = Path(args.teacher_checkpoint)
    teacher_nodes, teacher_channels = inspect_teacher_checkpoint(
        teacher_checkpoint,
        require_future_focused=False,
        expected_teacher_dim=128,
        expected_mask_policy="future_biased",
        expected_future_block_ratio=0.5,
        expected_mask_blocks=8,
    )
    args.data = args.berkeley_data
    args.student_distill_scope = "all"
    args.pipeline = "in_domain"
    datasets = datasets_for(args, args.berkeley_data)
    nodes, channels = inspect_dataset(datasets, "Berkeley")
    if channels != teacher_channels:
        raise ValueError("Teacher and Berkeley traffic channel counts differ")
    loaders = make_loaders(args, datasets)

    if args.only_c:
        c_path = train_c(
            args, loaders, teacher_checkpoint, teacher_nodes, nodes, channels,
            device, output,
        )
        checkpoints = {"C": c_path}
    else:
        a_path, b_path = train_a_b(
            args, loaders, teacher_checkpoint, teacher_nodes, nodes, channels,
            device, output,
        )
        c_path = train_c(
            args, loaders, teacher_checkpoint, teacher_nodes, nodes, channels,
            device, output,
        )
        checkpoints = {"A": a_path, "B": b_path, "C": c_path}
    results = {
        name: downstream_result(
            args, checkpoint, loaders, nodes, channels, device, output, name
        )
        for name, checkpoint in checkpoints.items()
    }
    summarize(args, output, checkpoints, results)


if __name__ == "__main__":
    main()
