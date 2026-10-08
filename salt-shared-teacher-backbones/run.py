#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from salt_backbones.data import load_city
from salt_backbones.graph import load_adjacency
from salt_backbones.models import BACKBONES, build_backbone, compatible_encoder_transfer
from salt_backbones.shared_teacher import build_shared_teacher
from salt_backbones.training import (
    fit_distillation, fit_forecast, make_loaders, seed_everything, write_json,
)


def arguments():
    parser = argparse.ArgumentParser(description="Shared-Teacher SALT on traffic Transformer encoders")
    parser.add_argument("--model", required=True, choices=BACKBONES)
    parser.add_argument(
        "--teacher-checkpoint", default="",
        help="One shared traffic Teacher checkpoint reused by every backbone",
    )
    parser.add_argument("--source-data", required=True, help="Oakland Teacher/Student source data")
    parser.add_argument("--target-data", required=True, help="Berkeley downstream data")
    parser.add_argument("--source-adj", required=True)
    parser.add_argument("--target-adj", required=True)
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
    parser.add_argument("--mask-blocks", type=int, default=8)
    parser.add_argument("--teacher-future-block-ratio", type=float, default=0.5)
    parser.add_argument("--mask-min-time", type=int, default=2)
    parser.add_argument("--mask-max-time", type=int, default=6)
    parser.add_argument("--mask-min-sensor-ratio", type=float, default=0.10)
    parser.add_argument("--mask-max-sensor-ratio", type=float, default=0.30)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip", default="", help="Comma-separated methods: scratch,salt,supervised")
    return parser.parse_args()


def append_summary(path: Path, row: dict):
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    identity = (
        row["model"], row["framework"], row["method"],
        str(row["seed"]), str(row["horizon"]),
    )
    rows = [old for old in rows if (
        old["model"], old["framework"], old["method"], old["seed"],
        old.get("horizon", str(row["horizon"])),
    ) != identity]
    rows.append({key: str(value) for key, value in row.items()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


def fresh(args, dataset, adjacency):
    return build_backbone(
        args.model, dataset.num_nodes, args.input_steps, args.pred_steps,
        args.dropout, adj=adjacency, train_x=dataset.history.numpy(),
    )


def load_shared_teacher(args, path, dataset, adjacency, device, output):
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu")
    if "teacher" not in saved:
        raise ValueError(f"{checkpoint} is not a shared traffic Teacher checkpoint")
    metadata = saved.get("metadata", {})
    required = {
        "architecture": "shared_spatiotemporal_graph_transformer_v1",
        "input_steps": args.input_steps,
        "pred_steps": args.pred_steps,
        "mask_blocks": args.mask_blocks,
        "future_block_ratio": args.teacher_future_block_ratio,
        "mask_min_time": args.mask_min_time,
        "mask_max_time": args.mask_max_time,
        "mask_min_sensor_ratio": args.mask_min_sensor_ratio,
        "mask_max_sensor_ratio": args.mask_max_sensor_ratio,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Teacher checkpoint {key}={metadata.get(key)!r}; expected {expected!r}"
            )
    if int(metadata.get("nodes", -1)) != dataset.num_nodes:
        raise ValueError(
            f"Shared Teacher has {metadata.get('nodes')} nodes but Student source "
            f"data has {dataset.num_nodes}; use the same source city for Stage 2"
        )
    teacher = build_shared_teacher(adjacency, metadata)
    current_graph = teacher.normalized_adjacency
    saved_graph = saved["teacher"].get("normalized_adjacency")
    if saved_graph is None or not torch.allclose(current_graph, saved_graph, atol=1e-6):
        raise ValueError("Shared Teacher checkpoint and source adjacency differ")
    teacher.load_state_dict(saved["teacher"])
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    write_json(output / "teacher" / "reuse_report.json", {
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "student_backbone": args.model,
        "student_source_nodes": dataset.num_nodes,
        "mode": "shared_exact_teacher",
    })
    return teacher, saved


def record(args, method, metrics, extra=None):
    row = {
        "model": args.model, "framework": "Shared-Teacher SALT", "method": method, **asdict(metrics),
        "val_latent": "", "teacher_val_reconstruction": "",
        "teacher_future_mask_fraction": "", "seed": args.seed,
        "history": args.input_steps, "horizon": args.pred_steps,
    }
    if extra:
        row.update(extra)
    append_summary(Path(args.output) / "summary.csv", row)
    print(json.dumps(row, indent=2), flush=True)


def main():
    args = arguments()
    if args.input_steps != 12 or args.pred_steps not in (6, 9, 12):
        raise ValueError("Supported protocol: 12 history -> horizon in {6, 9, 12}")
    if not 0 <= args.teacher_future_block_ratio <= 1:
        raise ValueError("--teacher-future-block-ratio must be in [0, 1]")
    skip = {name.strip() for name in args.skip.split(",") if name.strip()}
    invalid = skip - {"scratch", "salt", "supervised"}
    if invalid:
        raise ValueError(f"Unknown skipped methods: {invalid}")
    if "salt" not in skip and not args.teacher_checkpoint:
        raise ValueError(
            "Shared-Teacher SALT requires --teacher-checkpoint. Run "
            "train_shared_teacher.py once, then reuse that checkpoint for all backbones."
        )
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
    target_adj = load_adjacency(args.target_adj, nodes)
    print(f"model={args.model}; device={device}; target_nodes={nodes}; "
          f"sizes={ {key: len(value) for key, value in target.items()} }")

    source = source_loaders = source_adj = None
    if "salt" not in skip or "supervised" not in skip:
        source = load_city(
            args.source_data, args.input_steps, args.pred_steps,
            args.traffic_feature, args.file_pattern,
        )
        source_loaders = make_loaders(source, args.batch_size, args.workers)
        source_adj = load_adjacency(args.source_adj, source["train"].num_nodes)

    if "scratch" not in skip:
        seed_everything(args.seed)
        metrics = fit_forecast(
            fresh(args, target["train"], target_adj), target_loaders, device, output / "scratch",
            args.forecast_epochs, args.finetune_lr, 1.0, args.patience, "scratch",
        )
        record(args, "target scratch", metrics)

    if "salt" not in skip:
        assert source is not None and source_loaders is not None and source_adj is not None
        seed_everything(args.seed + 10)
        teacher, teacher_result = load_shared_teacher(
            args, args.teacher_checkpoint, source["train"], source_adj,
            device, output,
        )
        seed_everything(args.seed + 11)
        student = fresh(args, source["train"], source_adj)
        distilled = fit_distillation(
            student, teacher, source_loaders, device, output / "distillation",
            args.student_epochs, args.student_lr, args.patience,
            args.distill_loss, args.dropout,
        )
        latent = float(distilled["val_latent"])
        seed_everything(args.seed + 12)
        model = fresh(args, target["train"], target_adj)
        transfer = compatible_encoder_transfer(model, distilled["student"])
        write_json(output / "salt_full" / "transfer_report.json", transfer)
        metrics = fit_forecast(
            model, target_loaders, device, output / "salt_full",
            args.forecast_epochs, args.finetune_lr, args.encoder_lr_scale,
            args.patience, "full",
        )
        record(args, "Shared SALT -> full FT", metrics, {
            "val_latent": latent,
            "teacher_val_reconstruction": float(teacher_result["val_reconstruction"]),
            "teacher_future_mask_fraction": float(
                teacher_result["val_future_mask_fraction"]
            ),
        })

    if "supervised" not in skip:
        assert source is not None and source_loaders is not None and source_adj is not None
        seed_everything(args.seed + 3)
        source_model = fresh(args, source["train"], source_adj)
        fit_forecast(
            source_model, source_loaders, device, output / "supervised_source",
            args.source_epochs, args.finetune_lr, 1.0, args.patience,
            "supervised_source",
        )
        seed_everything(args.seed + 4)
        target_model = fresh(args, target["train"], target_adj)
        report = compatible_encoder_transfer(
            target_model, {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
        )
        write_json(output / "supervised_transfer" / "transfer_report.json", report)
        metrics = fit_forecast(
            target_model, target_loaders, device, output / "supervised_transfer",
            args.forecast_epochs, args.finetune_lr, args.encoder_lr_scale,
            args.patience, "supervised_transfer",
        )
        record(args, "supervised source -> target full FT", metrics)

    write_json(output / "config_salt_controls.json", vars(args))


if __name__ == "__main__":
    main()
