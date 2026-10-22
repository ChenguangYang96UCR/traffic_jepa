#!/usr/bin/env python3
"""Run backbone-specific SALT on the paper-table models.

Existing ordinary-transfer/JEPA baselines are intentionally not retrained.
Use summarize_salt_multicounty.py to merge this file's results with earlier CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from run import REPO, adjacency, city_root, city_slug
from salt_backbones.data import load_city
from salt_backbones.models import compatible_encoder_transfer
from salt_backbones.training import (
    SpatioTemporalBlockMasker,
    TopologyAwareSpatioTemporalBlockMasker,
    fit_distillation,
    fit_forecast,
    fit_masked_teacher,
    make_loaders,
    seed_everything,
    write_json,
)
from transfer_benchmark.salt_native import (
    SALT_TABLE_BACKBONES,
    build_salt_table_backbone,
)


def arguments():
    parser = argparse.ArgumentParser(
        description="Backbone-specific Teacher/Student SALT on paper-table models"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--county", required=True)
    parser.add_argument("--source-city", required=True)
    parser.add_argument("--target-cities", required=True)
    parser.add_argument("--backbones", default=",".join(SALT_TABLE_BACKBONES))
    parser.add_argument("--horizons", default="6,9,12")
    parser.add_argument("--output", required=True)
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--distill-loss", choices=("l1", "l2"), default="l2")
    parser.add_argument("--mask-mode", choices=("random_st", "topology_st"),
                        default="topology_st")
    parser.add_argument("--mask-blocks", type=int, default=8)
    parser.add_argument("--future-block-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--force-teacher", action="store_true")
    parser.add_argument("--force-student", action="store_true")
    parser.add_argument("--force-target", action="store_true")
    return parser.parse_args()


def parse_csv_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def append_result(path: Path, row: dict):
    fields = list(row)
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    identity_fields = ("county", "source_city", "city", "backbone", "horizon", "seed")
    identity = tuple(str(row[key]) for key in identity_fields)
    rows = [old for old in rows if tuple(old.get(key, "") for key in identity_fields) != identity]
    rows.append({key: str(value) for key, value in row.items()})
    rows.sort(key=lambda item: (
        item["county"], item["city"], int(item["horizon"]), item["backbone"],
    ))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def masker_for(args, graph, horizon):
    options = dict(
        num_blocks=args.mask_blocks, min_time=2, max_time=min(6, horizon),
        min_sensor_ratio=0.10, max_sensor_ratio=0.30,
        future_start=args.history, future_block_ratio=args.future_block_ratio,
    )
    if args.mask_mode == "topology_st":
        return TopologyAwareSpatioTemporalBlockMasker(graph, **options)
    return SpatioTemporalBlockMasker(**options)


def fresh(args, backbone, dataset, graph, horizon):
    return build_salt_table_backbone(
        backbone, REPO, dataset, graph, args.history, horizon, args.dropout,
    )


def distilled_student(args, backbone, backbone_index, source, loaders, graph,
                      horizon, device, root):
    checkpoint = root / "best_source_student.pt"
    if checkpoint.exists() and not args.force_student:
        saved = torch.load(checkpoint, map_location="cpu")
        required = {
            "backbone": backbone, "horizon": horizon,
            "source_city": args.source_city, "seed": args.seed,
            "teacher_type": "backbone_specific",
            "mask_mode": args.mask_mode,
            "mask_blocks": args.mask_blocks,
            "future_block_ratio": args.future_block_ratio,
        }
        for key, value in required.items():
            if saved.get(key) != value:
                raise ValueError(f"Cached Student {key}={saved.get(key)!r}; expected {value!r}")
        return saved
    seed_everything(args.seed + 10_000 + 100 * backbone_index + horizon)
    teacher = fresh(args, backbone, source["train"], graph, horizon)
    teacher_result = fit_masked_teacher(
        teacher, loaders, device, root / "teacher",
        args.teacher_epochs, args.teacher_lr, args.patience,
        masker_for(args, graph, horizon),
        args.seed + 70_000 + 100 * backbone_index + horizon,
        checkpoint_metadata={
            "teacher_type": "backbone_specific",
            "backbone": backbone, "county": args.county,
            "source_city": args.source_city, "history": args.history,
            "horizon": horizon, "mask_mode": args.mask_mode,
            "mask_blocks": args.mask_blocks,
            "future_block_ratio": args.future_block_ratio,
            "seed": args.seed,
        },
    )
    seed_everything(args.seed + 20_000 + 100 * backbone_index + horizon)
    student = fresh(args, backbone, source["train"], graph, horizon)
    result = fit_distillation(
        student, teacher, loaders, device, root / "distillation",
        args.student_epochs, args.student_lr, args.patience,
        args.distill_loss, args.dropout,
    )
    saved = {
        "backbone": backbone, "source_city": args.source_city,
        "county": args.county, "history": args.history,
        "horizon": horizon, "seed": args.seed,
        "teacher_type": "backbone_specific",
        "mask_mode": args.mask_mode, "mask_blocks": args.mask_blocks,
        "future_block_ratio": args.future_block_ratio,
        "state": result["student"],
        "val_latent": float(result["val_latent"]),
        "teacher_val_reconstruction": float(
            teacher_result["val_reconstruction"]
        ),
        "teacher_val_future_mask_fraction": float(
            teacher_result["val_future_mask_fraction"]
        ),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, checkpoint)
    return saved


def completed(path):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if {"mae", "mse", "rmse"} <= value.keys() else None


def main():
    args = arguments()
    # Downstream caches depend on upstream weights. Cascading invalidation
    # prevents a newly trained Teacher from being paired with stale Students
    # or target metrics.
    if args.force_teacher:
        args.force_student = True
    if args.force_student:
        args.force_target = True
    backbones = [name.lower() for name in parse_csv_list(args.backbones)]
    invalid = sorted(set(backbones) - set(SALT_TABLE_BACKBONES))
    if invalid:
        raise ValueError(f"Unknown backbones={invalid}; valid={SALT_TABLE_BACKBONES}")
    horizons = [int(value) for value in parse_csv_list(args.horizons)]
    if not horizons or any(value not in (6, 9, 12) for value in horizons):
        raise ValueError("--horizons must be a subset of 6,9,12")
    targets = parse_csv_list(args.target_cities)
    if len(targets) != 2:
        raise ValueError("The controlled protocol requires exactly two target cities")

    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_root = city_root(data_root, args.source_city)
    print(
        f"SALT-only: county={args.county}; source={args.source_city}; "
        f"targets={targets}; backbones={backbones}; device={device}", flush=True,
    )

    for horizon in horizons:
        source = load_city(
            source_root, args.history, horizon,
            args.traffic_feature, args.file_pattern,
        )
        source_loaders = make_loaders(source, args.batch_size, args.workers)
        source_graph = adjacency(source_root, source["train"].num_nodes)
        for backbone_index, backbone in enumerate(backbones):
            print(f"\n===== SALT backbone={backbone} horizon={horizon} =====", flush=True)
            student_root = (
                output / "source" / city_slug(args.source_city)
                / f"h{horizon}" / "backbones" / backbone
            )
            student = distilled_student(
                args, backbone, backbone_index, source, source_loaders,
                source_graph, horizon, device, student_root,
            )
            for target_index, city in enumerate(targets):
                target_root = city_root(data_root, city)
                target = load_city(
                    target_root, args.history, horizon,
                    args.traffic_feature, args.file_pattern,
                )
                target_loaders = make_loaders(target, args.batch_size, args.workers)
                target_graph = adjacency(target_root, target["train"].num_nodes)
                run_root = (
                    output / "targets" / city_slug(city) / f"h{horizon}"
                    / backbone / "salt_backbone_teacher_full"
                )
                metrics = None if args.force_target else completed(run_root / "metrics.json")
                if metrics is None:
                    seed_everything(
                        args.seed + 30_000 + 100 * backbone_index
                        + 10 * target_index + horizon
                    )
                    model = fresh(args, backbone, target["train"], target_graph, horizon)
                    report = compatible_encoder_transfer(model, student["state"])
                    write_json(run_root / "transfer_report.json", report)
                    result = fit_forecast(
                        model, target_loaders, device, run_root,
                        args.finetune_epochs, args.finetune_lr,
                        args.encoder_lr_scale, args.patience, "full",
                    )
                    metrics = asdict(result)
                    del model
                row = {
                    "county": args.county, "city": city,
                    "source_city": args.source_city,
                    "method": f"{backbone}_salt", "backbone": backbone,
                    "horizon": horizon, "history": args.history,
                    "mae": metrics["mae"], "mse": metrics["mse"],
                    "rmse": metrics["rmse"], "seed": args.seed,
                    "transfer": "salt_full_finetune",
                    "teacher_type": "backbone_specific",
                    "mask_mode": args.mask_mode, "mask_blocks": args.mask_blocks,
                    "future_block_ratio": args.future_block_ratio,
                    "val_latent": student["val_latent"],
                    "teacher_val_reconstruction": student[
                        "teacher_val_reconstruction"
                    ],
                }
                append_result(output / "salt_results.csv", row)
                print(json.dumps(row), flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    write_json(output / f"config_{city_slug(args.county)}.json", vars(args))
    print(f"SALT-only results: {output / 'salt_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
