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
    parser.add_argument("--pipeline", choices=("same_city", "cross_city"), required=True)
    parser.add_argument("--pretrain-data", required=True)
    parser.add_argument("--target-data", required=True)
    parser.add_argument("--mode", choices=("all", "teacher", "student", "downstream"), default="all")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--student-checkpoint", default="")
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--pred-steps", type=int, default=12)

    parser.add_argument("--teacher-input-dim", type=int, default=16)
    parser.add_argument("--teacher-step-dim", type=int, default=16)
    parser.add_argument("--teacher-sensor-dim", type=int, default=32)
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
    return SpatioTemporalBlockMasker(
        num_blocks=args.mask_blocks,
        min_time=args.mask_min_time,
        max_time=args.mask_max_time,
        min_sensor_ratio=args.mask_min_sensor_ratio,
        max_sensor_ratio=args.mask_max_sensor_ratio,
    )


def masked_reconstruction_loss(prediction, target, mask):
    return torch.nn.functional.smooth_l1_loss(prediction[mask], target[mask])


@torch.no_grad()
def evaluate_teacher(model, loader, masker, device, seed: int) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    for history, future, _ in loader:
        sequence = torch.cat((history, future), dim=1).to(device)
        mask = masker(*sequence.shape[:3], device, generator)
        prediction = model(sequence, mask)
        loss = masked_reconstruction_loss(prediction, sequence, mask)
        masked_values = int(mask.sum())
        total += float(loss) * masked_values
        count += masked_values
    return total / count


def train_teacher(args, loaders, nodes, channels, device, output: Path) -> Path:
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
        total, count = 0.0, 0
        for history, future, _ in loaders["train"]:
            sequence = torch.cat((history, future), dim=1).to(device)
            mask = masker(*sequence.shape[:3], device)
            prediction = model(sequence, mask)
            loss = masked_reconstruction_loss(prediction, sequence, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            masked_values = int(mask.sum())
            total += float(loss.detach()) * masked_values
            count += masked_values
        validation = evaluate_teacher(
            model, loaders["val"], masker, device, args.seed + 77_000
        )
        print(
            f"teacher {epoch:03d}/{args.teacher_epochs}: "
            f"train_reconstruction={total / count:.6f} "
            f"val_reconstruction={validation:.6f}"
        )
        if validation < best:
            best, stale = validation, 0
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "epoch": epoch,
                    "val_reconstruction": validation,
                    "args": vars(args),
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping teacher after epoch {epoch}")
                break
    return checkpoint


@torch.no_grad()
def evaluate_student(model, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for history, future, _ in loader:
        prediction, target = model(history.to(device), future.to(device))
        loss = torch.nn.functional.l1_loss(prediction, target)
        total += float(loss) * target.numel()
        count += target.numel()
    return total / count


def train_student(
    args, loaders, nodes, channels, teacher_checkpoint: Path, device, output: Path
) -> Path:
    seed_everything(args.seed + 1)
    teacher = make_teacher_encoder(args, nodes, channels)
    teacher_state = torch.load(teacher_checkpoint, map_location="cpu")
    teacher.load_state_dict(teacher_state["encoder"])
    student = make_encoder(args, nodes, channels)
    model = SALTDistiller(
        teacher, student, args.input_steps, args.pred_steps, args.dropout
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
            f"train_distillation={total / count:.6f} "
            f"val_distillation={validation:.6f}"
        )
        if validation < best:
            best, stale = validation, 0
            torch.save(
                {
                    "student_encoder": model.student.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "teacher_checkpoint": str(teacher_checkpoint),
                    "epoch": epoch,
                    "val_distillation": validation,
                    "args": vars(args),
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping student after epoch {epoch}")
                break
    return checkpoint


def run_downstream(
    args,
    student_checkpoint: Path,
    pretrain_nodes: int,
    target_datasets,
    target_nodes: int,
    target_channels: int,
    device,
    output: Path,
):
    target_loaders = make_loaders(args, target_datasets)
    student_state = torch.load(student_checkpoint, map_location="cpu")["student_encoder"]
    results = {}
    args.freeze_city_embedding = args.pipeline == "same_city"

    seed_everything(args.seed)
    scratch = make_encoder(args, target_nodes, target_channels)
    results["staeformer_scratch"] = train_downstream(
        args, target_loaders, scratch, "scratch", device, output
    )

    transfer_report = None
    for strategy in ("frozen", "full"):
        encoder = make_encoder(args, target_nodes, target_channels)
        if args.pipeline == "same_city":
            if pretrain_nodes != target_nodes:
                raise ValueError("same_city requires identical pretrain/target node counts")
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

    report = {
        "pipeline": args.pipeline,
        "pretrain_nodes": pretrain_nodes,
        "target_nodes": target_nodes,
        **transfer_report,
    }
    write_json(output / "student_load_report.json", report)
    labels = {
        "staeformer_scratch": "STAEformer scratch",
        "salt_frozen": "SALT student -> frozen",
        "salt_full": "SALT student -> full FT",
    }
    lines = [f"{'Method':<34} {'MAE':>12} {'MSE':>12} {'RMSE':>12}", "-" * 73]
    for key in ("staeformer_scratch", "salt_frozen", "salt_full"):
        item = results[key]
        lines.append(
            f"{labels[key]:<34} {item['mae']:>12.7f} {item['mse']:>12.7f} {item['rmse']:>12.7f}"
        )
    summary = "\n".join(lines)
    print(f"\nSALT {args.pipeline} evaluation\n{summary}")
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_json(output / "all_results.json", results)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    output = Path(args.output)

    if (
        args.pipeline == "same_city"
        and Path(args.pretrain_data).resolve() != Path(args.target_data).resolve()
    ):
        raise ValueError(
            "same_city requires --pretrain-data and --target-data to be the same directory"
        )

    pretrain_datasets = datasets_for(args, args.pretrain_data)
    pretrain_nodes, pretrain_channels = inspect_dataset(
        pretrain_datasets, "SALT pretrain city"
    )
    pretrain_loaders = make_loaders(args, pretrain_datasets)

    teacher_checkpoint = Path(args.teacher_checkpoint) if args.teacher_checkpoint else None
    if args.mode in ("all", "teacher") and teacher_checkpoint is None:
        teacher_checkpoint = train_teacher(
            args,
            pretrain_loaders,
            pretrain_nodes,
            pretrain_channels,
            device,
            output,
        )
    if args.mode == "teacher":
        print(f"Saved static teacher to {teacher_checkpoint.resolve()}")
        return
    if teacher_checkpoint is None and args.mode == "student":
        raise ValueError("--mode student requires --teacher-checkpoint")

    student_checkpoint = Path(args.student_checkpoint) if args.student_checkpoint else None
    if args.mode in ("all", "student") and student_checkpoint is None:
        if teacher_checkpoint is None:
            raise ValueError("Student training requires a teacher checkpoint")
        student_checkpoint = train_student(
            args,
            pretrain_loaders,
            pretrain_nodes,
            pretrain_channels,
            teacher_checkpoint,
            device,
            output,
        )
    if args.mode == "student":
        print(f"Saved SALT student to {student_checkpoint.resolve()}")
        return
    if student_checkpoint is None:
        raise ValueError("--mode downstream requires --student-checkpoint")

    target_datasets = datasets_for(args, args.target_data)
    target_nodes, target_channels = inspect_dataset(target_datasets, "downstream city")
    if pretrain_channels != target_channels:
        raise ValueError("Pretraining and downstream channel counts differ")
    run_downstream(
        args,
        student_checkpoint,
        pretrain_nodes,
        target_datasets,
        target_nodes,
        target_channels,
        device,
        output,
    )


if __name__ == "__main__":
    main()
