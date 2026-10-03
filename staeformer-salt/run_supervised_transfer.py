#!/usr/bin/env python3
"""Supervised STAEformer Oakland-to-Berkeley transfer baseline."""
from __future__ import annotations

import argparse
import copy
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
from staeformer_salt.training import seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--target-data", required=True)
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--pred-steps", type=int, default=12)

    # Match the SALT Student / downstream STAEformer architecture.
    parser.add_argument("--input-embedding-dim", type=int, default=24)
    parser.add_argument("--step-embedding-dim", type=int, default=24)
    parser.add_argument("--sensor-embedding-dim", type=int, default=80)
    parser.add_argument("--feed-forward-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--source-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--forecast-loss", choices=("mae", "mse"), default="mae")
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/oakland_supervised_to_berkeley")
    return parser.parse_args()


def encoder_state_from_forecast_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    saved = torch.load(path, map_location="cpu")
    model_state = saved.get("model", {})
    prefix = "encoder."
    encoder_state = {
        name[len(prefix):]: value
        for name, value in model_state.items()
        if name.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError(f"{path} contains no STAEformer encoder parameters")
    return encoder_state


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    output = Path(args.output)
    source_root = Path(args.source_data)
    target_root = Path(args.target_data)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not target_root.exists():
        raise FileNotFoundError(target_root)
    if source_root.resolve() == target_root.resolve():
        raise ValueError("Supervised transfer requires different source and target datasets")

    source_datasets = datasets_for(args, str(source_root))
    source_nodes, source_channels = inspect_dataset(source_datasets, "source city")
    source_args = copy.copy(args)
    source_args.finetune_epochs = args.source_epochs
    source_args.finetune_lr = args.source_lr
    source_args.freeze_city_embedding = False

    print("\nStage 1: supervised STAEformer forecasting on Oakland")
    source_encoder = make_encoder(source_args, source_nodes, source_channels)
    source_metrics = train_downstream(
        source_args,
        make_loaders(source_args, source_datasets),
        source_encoder,
        "scratch",
        device,
        output / "source_oakland",
    )
    source_checkpoint = output / "source_oakland" / "scratch" / "best_forecast.pt"
    source_encoder_state = encoder_state_from_forecast_checkpoint(source_checkpoint)

    target_datasets = datasets_for(args, str(target_root))
    target_nodes, target_channels = inspect_dataset(target_datasets, "target city")
    if source_channels != target_channels:
        raise ValueError("Source and target traffic channel counts differ")
    target_loaders = make_loaders(args, target_datasets)
    args.freeze_city_embedding = False

    print("\nStage 2: Berkeley scratch and Oakland-supervised encoder transfer")
    results = {"source_oakland_test": source_metrics}
    seed_everything(args.seed)
    scratch = make_encoder(args, target_nodes, target_channels)
    results["berkeley_scratch"] = train_downstream(
        args, target_loaders, scratch, "scratch", device, output / "target_berkeley"
    )

    transfer_reports = {}
    for strategy in ("frozen", "full"):
        encoder = make_encoder(args, target_nodes, target_channels)
        transfer_reports[strategy] = load_cross_city_encoder(
            encoder, source_encoder_state
        )
        results[f"supervised_{strategy}"] = train_downstream(
            args,
            target_loaders,
            encoder,
            strategy,
            device,
            output / "target_berkeley",
        )

    write_json(
        output / "transfer_report.json",
        {
            "source_data": str(source_root),
            "target_data": str(target_root),
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "source_checkpoint": str(source_checkpoint),
            "reports": transfer_reports,
        },
    )
    write_json(output / "all_results.json", results)

    labels = [
        ("Berkeley STAEformer scratch", "berkeley_scratch"),
        ("Oakland supervised -> frozen", "supervised_frozen"),
        ("Oakland supervised -> full FT", "supervised_full"),
    ]
    lines = [
        "Oakland supervised STAEformer -> Berkeley transfer",
        f"{'Method':<38} {'MAE':>12} {'MSE':>12} {'RMSE':>12}",
        "-" * 77,
    ]
    for label, key in labels:
        item = results[key]
        lines.append(
            f"{label:<38} {item['mae']:>12.7f} "
            f"{item['mse']:>12.7f} {item['rmse']:>12.7f}"
        )
    summary = "\n".join(lines)
    print(f"\n{summary}")
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
