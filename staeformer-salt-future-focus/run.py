#!/usr/bin/env python3
"""SALT teacher reconstruction, student distillation, and STAEformer evaluation."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from experiment import (
    datasets_for,
    inspect_dataset,
    load_cross_city_encoder,
    make_encoder,
    make_loaders,
    train_downstream,
)
from staeformer_salt.model import STAEformerEncoder
from staeformer_salt.distillation import (
    ReconstructionTeacher,
    SALTDistiller,
    SpatioTemporalBlockMasker,
)
from staeformer_salt.training import seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SALT distillation for STAEformer")
    parser.add_argument(
        "--pipeline",
        choices=("in_domain", "cross_city", "fremont_direct", "oakland_transfer"),
        default="in_domain",
        help=(
            "in_domain: student and downstream use the same city; cross_city: "
            "distill the student on one city and transfer it to another. The two "
            "legacy names remain accepted for reproducibility."
        ),
    )
    parser.add_argument("--teacher-data", default="", help="Teacher city dataset")
    parser.add_argument("--student-data", default="", help="Student city dataset")
    parser.add_argument("--target-data", default="", help="Downstream city dataset")
    parser.add_argument(
        "--experiment-label",
        default="",
        help="Human-readable label used in the evaluation report",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "teacher", "student", "student_downstream", "downstream"),
        default="all",
    )
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--student-checkpoint", default="")
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--pred-steps", type=int, default=12)

    parser.add_argument("--teacher-input-dim", type=int, default=16)
    parser.add_argument("--teacher-step-dim", type=int, default=16)
    parser.add_argument(
        "--teacher-sensor-dim",
        type=int,
        default=0,
        help="Keep at 0 for node-count-independent cross-city supervision",
    )
    parser.add_argument("--teacher-ff-dim", type=int, default=128)
    parser.add_argument("--teacher-heads", type=int, default=4)
    parser.add_argument("--teacher-layers", type=int, default=2)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)

    parser.add_argument("--input-embedding-dim", type=int, default=24)
    parser.add_argument("--step-embedding-dim", type=int, default=24)
    parser.add_argument("--sensor-embedding-dim", type=int, default=80)
    parser.add_argument("--feed-forward-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--student-epochs", type=int, default=50)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--mask-blocks", type=int, default=3)
    parser.add_argument("--mask-min-time", type=int, default=2)
    parser.add_argument("--mask-max-time", type=int, default=6)
    parser.add_argument("--mask-min-sensor-ratio", type=float, default=0.10)
    parser.add_argument("--mask-max-sensor-ratio", type=float, default=0.30)
    parser.add_argument(
        "--teacher-mask-policy",
        choices=("uniform", "future_biased"),
        default="uniform",
    )
    parser.add_argument(
        "--teacher-future-block-ratio",
        type=float,
        default=0.75,
        help="Fraction of Teacher mask blocks allocated wholly to future steps",
    )
    parser.add_argument(
        "--student-distill-scope",
        choices=("all", "future"),
        default="future",
        help="Compare all 24 latent steps or future latent steps only",
    )
    parser.add_argument(
        "--require-future-focused-teacher",
        action="store_true",
        help="Reject legacy Teacher checkpoints not trained with future-biased masking",
    )
    parser.add_argument(
        "--expected-teacher-dim",
        type=int,
        default=0,
        help="When positive, reject a Teacher checkpoint with a different model dimension",
    )

    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--forecast-loss", choices=("mae", "mse"), default="mae")
    parser.add_argument(
        "--downstream-strategies",
        default="scratch,frozen,full",
        help="Comma-separated subset of scratch,frozen,full",
    )
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/salt")
    return parser.parse_args()


def make_teacher_encoder(args, nodes: int, channels: int) -> STAEformerEncoder:
    return STAEformerEncoder(
        num_nodes=nodes,
        max_steps=args.input_steps + args.pred_steps,
        input_dim=channels,
        input_embedding_dim=args.teacher_input_dim,
        step_embedding_dim=args.teacher_step_dim,
        sensor_embedding_dim=args.teacher_sensor_dim,
        feed_forward_dim=args.teacher_ff_dim,
        num_heads=args.teacher_heads,
        num_layers=args.teacher_layers,
        dropout=args.dropout,
    )


def make_masker(args) -> SpatioTemporalBlockMasker:
    future_start = (
        args.input_steps if args.teacher_mask_policy == "future_biased" else None
    )
    return SpatioTemporalBlockMasker(
        num_blocks=args.mask_blocks,
        min_time=args.mask_min_time,
        max_time=args.mask_max_time,
        min_sensor_ratio=args.mask_min_sensor_ratio,
        max_sensor_ratio=args.mask_max_sensor_ratio,
        future_start=future_start,
        future_block_ratio=args.teacher_future_block_ratio,
    )


def masked_reconstruction_loss(prediction, target, mask):
    return torch.nn.functional.smooth_l1_loss(prediction[mask], target[mask])


def reconstruction_statistics(prediction, target, mask, input_steps: int):
    loss_map = torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction="none"
    )
    history_mask = mask.clone()
    history_mask[:, input_steps:] = False
    future_mask = mask.clone()
    future_mask[:, :input_steps] = False

    def aggregate(region_mask):
        values = loss_map[region_mask]
        return float(values.sum()), values.numel()

    total_sum, total_count = aggregate(mask)
    history_sum, history_count = aggregate(history_mask)
    future_sum, future_count = aggregate(future_mask)
    return {
        "total_sum": total_sum,
        "total_count": total_count,
        "history_sum": history_sum,
        "history_count": history_count,
        "future_sum": future_sum,
        "future_count": future_count,
    }


@torch.no_grad()
def evaluate_teacher(model, loader, masker, device, seed: int, input_steps: int):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {
        "total_sum": 0.0, "total_count": 0,
        "history_sum": 0.0, "history_count": 0,
        "future_sum": 0.0, "future_count": 0,
    }
    for history, future, _ in loader:
        sequence = torch.cat((history, future), dim=1).to(device)
        mask = masker(*sequence.shape[:3], device, generator)
        prediction = model(sequence, mask)
        batch_stats = reconstruction_statistics(
            prediction, sequence, mask, input_steps
        )
        for key, value in batch_stats.items():
            totals[key] += value
    return {
        "total": totals["total_sum"] / totals["total_count"],
        "history": (
            totals["history_sum"] / totals["history_count"]
            if totals["history_count"] else None
        ),
        "future": (
            totals["future_sum"] / totals["future_count"]
            if totals["future_count"] else None
        ),
        "future_mask_fraction": totals["future_count"] / totals["total_count"],
    }


def train_teacher(args, loaders, nodes, channels, device, output: Path) -> Path:
    if args.teacher_sensor_dim != 0:
        raise ValueError(
            "The controlled experiments require --teacher-sensor-dim 0 so every "
            "Teacher architecture is node-count independent"
        )
    seed_everything(args.seed)
    model = ReconstructionTeacher(make_teacher_encoder(args, nodes, channels)).to(device)
    masker = make_masker(args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.teacher_lr, weight_decay=args.weight_decay
    )
    checkpoint = output / "teacher" / "best_teacher.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, stale = math.inf, 0
    for epoch in range(1, args.teacher_epochs + 1):
        model.train()
        totals = {
            "total_sum": 0.0, "total_count": 0,
            "history_sum": 0.0, "history_count": 0,
            "future_sum": 0.0, "future_count": 0,
        }
        for history, future, _ in loaders["train"]:
            sequence = torch.cat((history, future), dim=1).to(device)
            mask = masker(*sequence.shape[:3], device)
            prediction = model(sequence, mask)
            loss = masked_reconstruction_loss(prediction, sequence, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_stats = reconstruction_statistics(
                prediction.detach(), sequence, mask, args.input_steps
            )
            for key, value in batch_stats.items():
                totals[key] += value
        validation = evaluate_teacher(
            model, loaders["val"], masker, device, args.seed + 77_000,
            args.input_steps,
        )
        train_loss = totals["total_sum"] / totals["total_count"]
        train_future_fraction = totals["future_count"] / totals["total_count"]
        print(
            f"teacher {epoch:03d}/{args.teacher_epochs}: "
            f"train_reconstruction={train_loss:.6f} "
            f"val_reconstruction={validation['total']:.6f} "
            f"val_history={validation['history']} "
            f"val_future={validation['future']} "
            f"train_future_mask_fraction={train_future_fraction:.3f} "
            f"val_future_mask_fraction={validation['future_mask_fraction']:.3f}"
        )
        if validation["total"] < best:
            best, stale = validation["total"], 0
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "epoch": epoch,
                    "val_reconstruction": validation["total"],
                    "teacher_validation": validation,
                    "args": vars(args),
                    "teacher_nodes": nodes,
                    "teacher_channels": channels,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping teacher after epoch {epoch}")
                break
    best_state = torch.load(checkpoint, map_location="cpu")
    write_json(
        output / "teacher" / "teacher_metrics.json",
        {
            "mask_policy": args.teacher_mask_policy,
            "future_block_ratio": args.teacher_future_block_ratio,
            "mask_blocks": args.mask_blocks,
            "best_epoch": best_state["epoch"],
            "validation": best_state.get("teacher_validation", {}),
        },
    )
    return checkpoint


@torch.no_grad()
def evaluate_student(model, loader, device) -> dict[str, float]:
    model.eval()
    sums = {"scope": 0.0, "all": 0.0, "history": 0.0, "future": 0.0}
    counts = {"scope": 0, "all": 0, "history": 0, "future": 0}
    for history, future, _ in loader:
        predicted_all, target_all = model.latent_pairs(
            history.to(device), future.to(device)
        )
        predicted_scope, target_scope = model.select_scope(
            predicted_all, target_all
        )
        regions = {
            "scope": (predicted_scope, target_scope),
            "all": (predicted_all, target_all),
            "history": (
                predicted_all[:, :model.input_steps],
                target_all[:, :model.input_steps],
            ),
            "future": (
                predicted_all[:, model.input_steps:],
                target_all[:, model.input_steps:],
            ),
        }
        for name, (prediction, target) in regions.items():
            sums[name] += float((prediction - target).abs().sum())
            counts[name] += target.numel()
    return {name: sums[name] / counts[name] for name in sums}


def train_student(
    args, loaders, teacher_nodes, student_nodes, channels,
    teacher_checkpoint: Path, device, output: Path
) -> Path:
    seed_everything(args.seed + 1)
    teacher_state = torch.load(teacher_checkpoint, map_location="cpu")
    saved_args = teacher_state.get("args", {})
    teacher_max_steps = int(
        teacher_state["encoder"]["relative_step_embedding"].shape[0]
    )
    teacher = STAEformerEncoder(
        num_nodes=teacher_nodes,
        max_steps=teacher_max_steps,
        input_dim=channels,
        input_embedding_dim=saved_args.get("teacher_input_dim", args.teacher_input_dim),
        step_embedding_dim=saved_args.get("teacher_step_dim", args.teacher_step_dim),
        sensor_embedding_dim=saved_args.get("teacher_sensor_dim", args.teacher_sensor_dim),
        feed_forward_dim=saved_args.get("teacher_ff_dim", args.teacher_ff_dim),
        num_heads=saved_args.get("teacher_heads", args.teacher_heads),
        num_layers=saved_args.get("teacher_layers", args.teacher_layers),
        dropout=saved_args.get("dropout", args.dropout),
    )
    teacher.load_state_dict(teacher_state["encoder"])
    student = make_encoder(args, student_nodes, channels)
    model = SALTDistiller(
        teacher,
        student,
        args.input_steps,
        args.pred_steps,
        args.dropout,
        distill_scope=args.student_distill_scope,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.student.parameters()) + list(model.predictor.parameters()),
        lr=args.student_lr,
        weight_decay=args.weight_decay,
    )
    checkpoint = output / "student" / "best_student.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, stale = math.inf, 0
    for epoch in range(1, args.student_epochs + 1):
        model.train()
        model.teacher.eval()
        total, count = 0.0, 0
        for history, future, _ in loaders["train"]:
            prediction, target = model(history.to(device), future.to(device))
            loss = torch.nn.functional.l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = list(model.student.parameters()) + list(model.predictor.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += float(loss.detach()) * target.numel()
            count += target.numel()
        validation = evaluate_student(model, loaders["val"], device)
        print(
            f"student {epoch:03d}/{args.student_epochs}: "
            f"scope={args.student_distill_scope} "
            f"train_distillation={total / count:.6f} "
            f"val_scope={validation['scope']:.6f} "
            f"val_all={validation['all']:.6f} "
            f"val_history={validation['history']:.6f} "
            f"val_future={validation['future']:.6f}"
        )
        if validation["scope"] < best:
            best, stale = validation["scope"], 0
            torch.save(
                {
                    "student_encoder": model.student.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "teacher_checkpoint": str(teacher_checkpoint),
                    "epoch": epoch,
                    "val_distillation": validation["scope"],
                    "student_validation": validation,
                    "args": vars(args),
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping student after epoch {epoch}")
                break
    best_state = torch.load(checkpoint, map_location="cpu")
    write_json(
        output / "student" / "distillation_metrics.json",
        {
            "distill_scope": args.student_distill_scope,
            "best_epoch": best_state["epoch"],
            "validation": best_state.get("student_validation", {}),
            "teacher_checkpoint": str(teacher_checkpoint),
        },
    )
    return checkpoint


def run_downstream(
    args,
    student_checkpoint: Path,
    student_nodes: int,
    target_datasets,
    target_nodes: int,
    target_channels: int,
    device,
    output: Path,
):
    target_loaders = make_loaders(args, target_datasets)
    student_state = torch.load(student_checkpoint, map_location="cpu")["student_encoder"]
    results = {}
    strategies = [
        value.strip() for value in args.downstream_strategies.split(",")
        if value.strip()
    ]
    if not strategies or len(strategies) != len(set(strategies)):
        raise ValueError("--downstream-strategies must contain unique strategies")
    invalid = sorted(set(strategies) - {"scratch", "frozen", "full"})
    if invalid:
        raise ValueError(f"Unknown downstream strategies: {invalid}")
    in_domain = args.pipeline in ("in_domain", "fremont_direct")
    args.freeze_city_embedding = in_domain

    if "scratch" in strategies:
        seed_everything(args.seed)
        scratch = make_encoder(args, target_nodes, target_channels)
        results["staeformer_scratch"] = train_downstream(
            args, target_loaders, scratch, "scratch", device, output
        )

    transfer_report = None
    for strategy in (value for value in ("frozen", "full") if value in strategies):
        encoder = make_encoder(args, target_nodes, target_channels)
        if in_domain:
            if student_nodes != target_nodes:
                raise ValueError("in_domain requires identical student/target node counts")
            encoder.load_state_dict(student_state)
            transfer_report = {
                "loaded": list(student_state),
                "reinitialized_city_specific": [],
                "shape_mismatch": [],
                "unexpected": [],
            }
        else:
            transfer_report = load_cross_city_encoder(encoder, student_state)
        results[f"salt_{strategy}"] = train_downstream(
            args, target_loaders, encoder, strategy, device, output
        )

    transfer_report = transfer_report or {
        "loaded": [],
        "reinitialized_city_specific": [],
        "shape_mismatch": [],
        "unexpected": [],
    }
    report = {
        "pipeline": args.pipeline,
        "experiment_label": args.experiment_label,
        "student_nodes": student_nodes,
        "target_nodes": target_nodes,
        "student_distill_scope": args.student_distill_scope,
        "downstream_strategies": strategies,
        **transfer_report,
    }
    write_json(output / "student_load_report.json", report)
    labels = {
        "staeformer_scratch": "STAEformer scratch",
        "salt_frozen": "SALT student -> frozen",
        "salt_full": "SALT student -> full FT",
    }
    lines = [f"{'Method':<34} {'MAE':>12} {'MSE':>12} {'RMSE':>12}", "-" * 73]
    result_order = [
        key for key in ("staeformer_scratch", "salt_frozen", "salt_full")
        if key in results
    ]
    for key in result_order:
        item = results[key]
        lines.append(
            f"{labels[key]:<34} {item['mae']:>12.7f} {item['mse']:>12.7f} {item['rmse']:>12.7f}"
        )
    summary = "\n".join(lines)
    title = args.experiment_label or f"SALT {args.pipeline} evaluation"
    print(f"\n{title}\n{summary}")
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_json(output / "all_results.json", results)


def require_path(value: str, option: str) -> Path:
    if not value:
        raise ValueError(f"{option} is required for this mode")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def inspect_teacher_checkpoint(
    path: Path,
    require_future_focused: bool = False,
    expected_teacher_dim: int = 0,
) -> tuple[int, int]:
    saved = torch.load(path, map_location="cpu")
    encoder = saved.get("encoder", {})
    if "input_projection.weight" not in encoder:
        raise ValueError(f"{path} is not a valid SALT Teacher checkpoint")
    if "sensor_embedding" in encoder:
        raise ValueError(
            f"{path} contains a city-specific sensor embedding. Retrain the shared "
            "Teacher with --teacher-sensor-dim 0."
        )
    nodes = int(saved.get("teacher_nodes", 0))
    if nodes <= 0:
        raise ValueError(f"{path} does not record the Teacher node count")
    channels = int(encoder["input_projection.weight"].shape[1])
    teacher_dim = int(encoder["input_projection.weight"].shape[0])
    if "relative_step_embedding" in encoder:
        teacher_dim += int(encoder["relative_step_embedding"].shape[1])
    if "sensor_embedding" in encoder:
        teacher_dim += int(encoder["sensor_embedding"].shape[1])
    if expected_teacher_dim > 0 and teacher_dim != expected_teacher_dim:
        raise ValueError(
            f"{path} has Teacher model dimension {teacher_dim}; "
            f"expected {expected_teacher_dim}."
        )
    saved_args = saved.get("args", {})
    if require_future_focused:
        policy = saved_args.get("teacher_mask_policy")
        ratio = float(saved_args.get("teacher_future_block_ratio", 0.0))
        if policy != "future_biased" or ratio <= 0.5:
            raise ValueError(
                f"{path} is not a future-focused Teacher checkpoint: "
                f"policy={policy!r}, future_block_ratio={ratio}. Retrain the "
                "Teacher with --teacher-mask-policy future_biased and a ratio > 0.5."
            )
    print(
        f"Frozen Teacher checkpoint: {path}; training_nodes={nodes}; "
        f"channels={channels}; model_dim={teacher_dim}; node-agnostic"
    )
    return nodes, channels


def main() -> None:
    args = parse_args()
    if (
        args.teacher_mask_policy == "future_biased"
        and args.teacher_future_block_ratio <= 0.5
    ):
        raise ValueError(
            "future_biased masking requires --teacher-future-block-ratio > 0.5"
        )
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    output = Path(args.output)

    teacher_checkpoint = Path(args.teacher_checkpoint) if args.teacher_checkpoint else None
    teacher_nodes = teacher_channels = None
    if args.mode in ("all", "teacher") and teacher_checkpoint is None:
        teacher_root = require_path(args.teacher_data, "--teacher-data")
        teacher_datasets = datasets_for(args, str(teacher_root))
        teacher_nodes, teacher_channels = inspect_dataset(
            teacher_datasets, "Teacher city"
        )
        teacher_checkpoint = train_teacher(
            args,
            make_loaders(args, teacher_datasets),
            teacher_nodes,
            teacher_channels,
            device,
            output,
        )
    if args.mode == "teacher":
        print(f"Saved static teacher to {teacher_checkpoint.resolve()}")
        return
    if teacher_checkpoint is None and args.mode in ("student", "student_downstream"):
        raise ValueError("Student training requires --teacher-checkpoint")

    student_checkpoint = Path(args.student_checkpoint) if args.student_checkpoint else None
    student_nodes = student_channels = None
    student_datasets = None
    if args.mode in ("all", "student", "student_downstream") and student_checkpoint is None:
        if teacher_checkpoint is None:
            raise ValueError("Student training requires a teacher checkpoint")
        student_root = require_path(args.student_data, "--student-data")
        if teacher_nodes is None:
            teacher_nodes, teacher_channels = inspect_teacher_checkpoint(
                teacher_checkpoint,
                args.require_future_focused_teacher,
                args.expected_teacher_dim,
            )
        student_datasets = datasets_for(args, str(student_root))
        student_nodes, student_channels = inspect_dataset(
            student_datasets, "SALT student city"
        )
        if teacher_channels != student_channels:
            raise ValueError("Teacher and student traffic channel counts differ")
        print("Frozen Teacher is node-agnostic; no sensor mapping is required")
        student_checkpoint = train_student(
            args,
            make_loaders(args, student_datasets),
            teacher_nodes,
            student_nodes,
            student_channels,
            teacher_checkpoint,
            device,
            output,
        )
    if args.mode == "student":
        print(f"Saved SALT student to {student_checkpoint.resolve()}")
        return
    if student_checkpoint is None:
        raise ValueError("--mode downstream requires --student-checkpoint")

    student_root = require_path(args.student_data, "--student-data")
    if student_datasets is None:
        student_datasets = datasets_for(args, str(student_root))
        student_nodes, student_channels = inspect_dataset(
            student_datasets, "SALT student city"
        )
    target_root = require_path(args.target_data, "--target-data")
    in_domain = args.pipeline in ("in_domain", "fremont_direct")
    if in_domain and student_root.resolve() != target_root.resolve():
        raise ValueError("in_domain requires the same --student-data and --target-data")
    if not in_domain and student_root.resolve() == target_root.resolve():
        raise ValueError("cross_city requires different student and target datasets")
    target_datasets = datasets_for(args, str(target_root))
    target_nodes, target_channels = inspect_dataset(target_datasets, "downstream city")
    if student_channels != target_channels:
        raise ValueError("Student and downstream channel counts differ")
    run_downstream(
        args,
        student_checkpoint,
        student_nodes,
        target_datasets,
        target_nodes,
        target_channels,
        device,
        output,
    )


if __name__ == "__main__":
    main()
