#!/usr/bin/env python3
"""Traffic-SALT masking ablation across counties, cities, and horizons."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from run import adjacency, build, city_root, city_slug, load_transfer
from salt_backbones.data import load_city
from salt_backbones.training import (
    MatchedRandomPointMasker,
    SpatioTemporalBlockMasker,
    TopologyAwareSpatioTemporalBlockMasker,
    fit_distillation,
    fit_forecast,
    fit_masked_teacher,
    make_loaders,
    seed_everything,
    write_json,
)


RATIOS = ((1, 9), (3, 7), (5, 5), (8, 2))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--county", required=True)
    parser.add_argument("--source-city", required=True)
    parser.add_argument("--target-cities", required=True)
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
    parser.add_argument("--mask-blocks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--force-source", action="store_true")
    parser.add_argument("--force-target", action="store_true")
    return parser.parse_args()


def configurations():
    configs = []
    for history_parts, future_parts in RATIOS:
        future_ratio = future_parts / (history_parts + future_parts)
        ratio = f"{history_parts}:{future_parts}"
        configs.extend((
            {"mask_mode": "random_block", "history_future_ratio": ratio,
             "future_block_ratio": future_ratio},
            {"mask_mode": "topology_block", "history_future_ratio": ratio,
             "future_block_ratio": future_ratio},
        ))
    configs.append({
        "mask_mode": "random_point", "history_future_ratio": "n/a",
        "future_block_ratio": None,
    })
    return configs


def config_slug(config):
    if config["mask_mode"] == "random_point":
        return "random_point"
    return f"{config['mask_mode']}_h{config['history_future_ratio'].replace(':', '_f')}"


def make_masker(args, graph, horizon, config):
    common = dict(
        num_blocks=args.mask_blocks, min_time=2, max_time=min(6, horizon),
        min_sensor_ratio=0.10, max_sensor_ratio=0.30,
        future_start=args.history,
        probabilistic_region_ratio=True,
    )
    if config["mask_mode"] == "random_point":
        # Only the masked-count distribution comes from the reference. Final
        # points are uniformly redrawn across all T x N positions.
        reference = SpatioTemporalBlockMasker(
            **common, future_block_ratio=0.5,
        )
        return MatchedRandomPointMasker(reference)
    options = {**common, "future_block_ratio": config["future_block_ratio"]}
    if config["mask_mode"] == "topology_block":
        return TopologyAwareSpatioTemporalBlockMasker(graph, **options)
    return SpatioTemporalBlockMasker(**options)


def append_result(path, row):
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    keys = (
        "county", "source_city", "city", "horizon", "mask_mode",
        "history_future_ratio", "seed",
    )
    identity = tuple(str(row[key]) for key in keys)
    rows = [old for old in rows if tuple(old.get(key, "") for key in keys) != identity]
    rows.append({key: str(value) for key, value in row.items()})
    rows.sort(key=lambda item: (
        item["county"], item["city"], int(item["horizon"]),
        item["mask_mode"], item["history_future_ratio"],
    ))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


def load_metrics(path):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if {"mae", "mse", "rmse"} <= value.keys() else None


def train_source(args, source, loaders, graph, horizon, config, device, root):
    checkpoint = root / "source_student.pt"
    if checkpoint.exists() and not args.force_source:
        saved = torch.load(checkpoint, map_location="cpu")
        expected = {
            "mask_mode": config["mask_mode"],
            "history_future_ratio": config["history_future_ratio"],
            "horizon": horizon, "source_city": args.source_city,
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"Cached source {key}={saved.get(key)!r}; expected {value!r}")
        return saved

    masker = make_masker(args, graph, horizon, config)
    seed_everything(args.seed + 10_000 + horizon)
    teacher = build("ours", source["train"], graph, args, horizon)
    teacher_result = fit_masked_teacher(
        teacher, loaders, device, root / "teacher",
        args.teacher_epochs, args.teacher_lr, args.patience,
        masker, args.seed + 70_000 + horizon,
        checkpoint_metadata={
            "county": args.county, "source_city": args.source_city,
            "horizon": horizon, "mask_mode": config["mask_mode"],
            "history_future_ratio": config["history_future_ratio"],
            "future_block_ratio": config["future_block_ratio"],
            "mask_blocks": args.mask_blocks,
        },
    )
    seed_everything(args.seed + 20_000 + horizon)
    student = build("ours", source["train"], graph, args, horizon)
    student_result = fit_distillation(
        student, teacher, loaders, device, root / "student",
        args.student_epochs, args.student_lr, args.patience,
        "l2", 0.1,
    )
    saved = {
        "method": "ours", "source_city": args.source_city,
        "county": args.county, "horizon": horizon,
        "mask_mode": config["mask_mode"],
        "history_future_ratio": config["history_future_ratio"],
        "future_block_ratio": config["future_block_ratio"],
        "mask_blocks": args.mask_blocks, "state": student_result["student"],
        "val_latent": float(student_result["val_latent"]),
        "teacher_val_reconstruction": float(teacher_result["val_reconstruction"]),
        "observed_future_mask_fraction": float(
            teacher_result["val_future_mask_fraction"]
        ),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, checkpoint)
    return saved


def main():
    args = arguments()
    if args.force_source:
        args.force_target = True
    targets = [city.strip() for city in args.target_cities.split(",") if city.strip()]
    if len(targets) != 2:
        raise ValueError("Exactly two target cities are required per county")
    horizons = [int(value.strip()) for value in args.horizons.split(",") if value.strip()]
    if not horizons or any(value not in (6, 9, 12) for value in horizons):
        raise ValueError("--horizons must be a subset of 6,9,12")

    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_root = city_root(data_root, args.source_city)
    print(
        f"mask ablation: county={args.county}; source={args.source_city}; "
        f"targets={targets}; horizons={horizons}; device={device}", flush=True,
    )

    for horizon in horizons:
        source = load_city(
            source_root, args.history, horizon,
            args.traffic_feature, args.file_pattern,
        )
        source_loaders = make_loaders(source, args.batch_size, args.workers)
        source_graph = adjacency(source_root, source["train"].num_nodes)
        for config_index, config in enumerate(configurations()):
            slug = config_slug(config)
            print(f"\n===== horizon={horizon} mask={slug} =====", flush=True)
            source_dir = output / "source" / city_slug(args.source_city) / f"h{horizon}" / slug
            pretrained = train_source(
                args, source, source_loaders, source_graph,
                horizon, config, device, source_dir,
            )
            for target_index, city in enumerate(targets):
                target_root = city_root(data_root, city)
                run_dir = output / "targets" / city_slug(city) / f"h{horizon}" / slug
                metrics = None if args.force_target else load_metrics(run_dir / "metrics.json")
                if metrics is None:
                    target = load_city(
                        target_root, args.history, horizon,
                        args.traffic_feature, args.file_pattern,
                    )
                    loaders = make_loaders(target, args.batch_size, args.workers)
                    target_graph = adjacency(target_root, target["train"].num_nodes)
                    # Same initialization seed across mask configurations.
                    seed_everything(args.seed + 30_000 + 10 * target_index + horizon)
                    model = build("ours", target["train"], target_graph, args, horizon)
                    report = load_transfer(model, "ours", pretrained)
                    write_json(run_dir / "transfer_report.json", report)
                    result = fit_forecast(
                        model, loaders, device, run_dir,
                        args.finetune_epochs, args.finetune_lr,
                        args.encoder_lr_scale, args.patience,
                        "supervised_transfer",
                    )
                    metrics = asdict(result)
                    del model
                row = {
                    "county": args.county, "source_city": args.source_city,
                    "city": city, "method": "ours", "horizon": horizon,
                    "history": args.history, "mask_mode": config["mask_mode"],
                    "history_future_ratio": config["history_future_ratio"],
                    "future_block_ratio": (
                        "n/a" if config["future_block_ratio"] is None
                        else config["future_block_ratio"]
                    ),
                    "mask_blocks": (
                        "n/a" if config["mask_mode"] == "random_point"
                        else args.mask_blocks
                    ),
                    "mae": metrics["mae"], "mse": metrics["mse"],
                    "rmse": metrics["rmse"], "seed": args.seed,
                    "transfer": "salt_full_finetune",
                    "val_latent": pretrained["val_latent"],
                    "teacher_val_reconstruction": pretrained["teacher_val_reconstruction"],
                    "observed_future_mask_fraction": pretrained[
                        "observed_future_mask_fraction"
                    ],
                }
                append_result(output / "mask_ablation_results.csv", row)
                print(json.dumps(row), flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    write_json(output / f"config_{city_slug(args.county)}.json", vars(args))


if __name__ == "__main__":
    main()
