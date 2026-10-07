#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from salt_backbones.data import load_city
from salt_backbones.models import build_backbone, compatible_encoder_transfer
from salt_backbones.teacher import load_frozen_teacher
from salt_backbones.training import (
    fit_distillation, fit_forecast, make_loaders, seed_everything, write_json,
)


def arguments():
    parser = argparse.ArgumentParser(description="SALT on transferable traffic Transformer encoders")
    parser.add_argument("--model", required=True, choices=("patchtst", "stgformer", "stlaformer"))
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--expected-teacher-dim", type=int, default=128)
    parser.add_argument("--source-data", required=True, help="Oakland for supervised-transfer control")
    parser.add_argument("--target-data", required=True, help="Berkeley student/downstream data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--pred-steps", type=int, default=12)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--forecast-epochs", type=int, default=100)
    parser.add_argument("--source-epochs", type=int, default=100)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--distill-loss", choices=("l1", "l2"), default="l2")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip", default="", help="Comma-separated methods: scratch,frozen,full,supervised")
    return parser.parse_args()


def append_summary(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def fresh(args, nodes):
    return build_backbone(args.model, nodes, args.input_steps, args.pred_steps, args.dropout)


def record(args, method, metrics, extra=None):
    row = {
        "model": args.model, "method": method, **asdict(metrics),
        "val_latent": "", "seed": args.seed,
    }
    if extra:
        row.update(extra)
    append_summary(Path(args.output) / "summary.csv", row)
    print(json.dumps(row, indent=2), flush=True)


def main():
    args = arguments()
    if (args.input_steps, args.pred_steps) != (12, 12):
        raise ValueError("This controlled comparison is fixed to 12 history -> 12 future")
    skip = {name.strip() for name in args.skip.split(",") if name.strip()}
    invalid = skip - {"scratch", "frozen", "full", "supervised"}
    if invalid:
        raise ValueError(f"Unknown skipped methods: {invalid}")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = load_city(
        args.target_data, args.input_steps, args.pred_steps,
        args.traffic_feature, args.file_pattern,
    )
    target_loaders = make_loaders(target, args.batch_size, args.workers)
    nodes = target["train"].num_nodes
    print(f"model={args.model}; device={device}; target_nodes={nodes}; "
          f"sizes={ {key: len(value) for key, value in target.items()} }")

    if "scratch" not in skip:
        seed_everything(args.seed)
        metrics = fit_forecast(
            fresh(args, nodes), target_loaders, device, output / "scratch",
            args.forecast_epochs, args.finetune_lr, 1.0, args.patience, "scratch",
        )
        record(args, "target scratch", metrics)

    teacher = load_frozen_teacher(args.teacher_checkpoint, device)
    if args.expected_teacher_dim and teacher.model_dim != args.expected_teacher_dim:
        raise ValueError(
            f"Teacher dimension is {teacher.model_dim}, expected {args.expected_teacher_dim}"
        )
    student = fresh(args, nodes)
    seed_everything(args.seed + 1)
    distilled = fit_distillation(
        student, teacher, target_loaders, device, output / "distillation",
        args.student_epochs, args.student_lr, args.patience,
        args.distill_loss, args.dropout,
    )
    latent = float(distilled["val_latent"])

    for strategy in ("frozen", "full"):
        if strategy in skip:
            continue
        seed_everything(args.seed + 2)
        model = fresh(args, nodes)
        model.load_state_dict(distilled["student"])
        metrics = fit_forecast(
            model, target_loaders, device, output / f"salt_{strategy}",
            args.forecast_epochs, args.finetune_lr, args.encoder_lr_scale,
            args.patience, strategy,
        )
        record(args, f"SALT -> {strategy}", metrics, {"val_latent": latent})

    if "supervised" not in skip:
        source = load_city(
            args.source_data, args.input_steps, args.pred_steps,
            args.traffic_feature, args.file_pattern,
        )
        source_loaders = make_loaders(source, args.batch_size, args.workers)
        seed_everything(args.seed + 3)
        source_model = fresh(args, source["train"].num_nodes)
        fit_forecast(
            source_model, source_loaders, device, output / "supervised_source",
            args.source_epochs, args.finetune_lr, 1.0, args.patience,
            "supervised_source",
        )
        target_model = fresh(args, nodes)
        report = compatible_encoder_transfer(
            target_model, {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
        )
        write_json(output / "supervised_transfer" / "transfer_report.json", report)
        seed_everything(args.seed + 4)
        metrics = fit_forecast(
            target_model, target_loaders, device, output / "supervised_transfer",
            args.forecast_epochs, args.finetune_lr, args.encoder_lr_scale,
            args.patience, "supervised_transfer",
        )
        record(args, "supervised source -> target full FT", metrics)

    write_json(output / "config.json", vars(args))


if __name__ == "__main__":
    main()
