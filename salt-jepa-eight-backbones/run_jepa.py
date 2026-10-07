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
from salt_backbones.models import BACKBONES, build_backbone
from salt_backbones.training import (
    fit_forecast, fit_jepa, make_loaders, seed_everything, write_json,
)


def arguments():
    parser = argparse.ArgumentParser(description="JEPA pretraining for traffic Transformer encoders")
    parser.add_argument("--model", required=True, choices=BACKBONES)
    parser.add_argument("--target-data", required=True)
    parser.add_argument("--target-adj", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--pred-steps", type=int, default=12)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--forecast-epochs", type=int, default=100)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--latent-loss", choices=("l1", "l2"), default="l2")
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def append(path: Path, row: dict):
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


def record(args, method, metrics, latent=""):
    row = {
        "model": args.model, "framework": "JEPA", "method": method,
        **asdict(metrics), "val_latent": latent, "seed": args.seed,
        "history": args.input_steps, "horizon": args.pred_steps,
    }
    append(Path(args.output) / "summary.csv", row)
    print(json.dumps(row, indent=2), flush=True)


def fresh(args, dataset, adjacency):
    return build_backbone(
        args.model, dataset.num_nodes, args.input_steps, args.pred_steps,
        args.dropout, adj=adjacency, train_x=dataset.history.numpy(),
    )


def main():
    args = arguments()
    if args.input_steps != 12 or args.pred_steps not in (6, 9, 12):
        raise ValueError("Supported protocol: 12 history -> horizon in {6, 9, 12}")
    if not 0.0 <= args.ema_momentum < 1.0:
        raise ValueError("EMA momentum must be in [0, 1)")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = load_city(
        args.target_data, args.input_steps, args.pred_steps,
        args.traffic_feature, args.file_pattern,
    )
    loaders = make_loaders(datasets, args.batch_size, args.workers)
    nodes = datasets["train"].num_nodes
    adjacency = load_adjacency(args.target_adj, nodes)
    print(f"JEPA model={args.model}; device={device}; nodes={nodes}; "
          f"sizes={ {key: len(value) for key, value in datasets.items()} }")

    seed_everything(args.seed + 1)
    online = fresh(args, datasets["train"], adjacency)
    saved = fit_jepa(
        online, loaders, device, output / "pretrain",
        args.pretrain_epochs, args.pretrain_lr, args.patience,
        args.latent_loss, args.dropout, args.ema_momentum,
    )
    latent = float(saved["val_latent"])
    seed_everything(args.seed + 2)
    model = fresh(args, datasets["train"], adjacency)
    # Evaluate the smoother EMA Target Encoder. The temporary JEPA Predictor
    # is discarded; the native forecast head and encoder are fully fine-tuned.
    model.load_state_dict(saved["target"])
    metrics = fit_forecast(
        model, loaders, device, output / "jepa_full",
        args.forecast_epochs, args.finetune_lr,
        args.encoder_lr_scale, args.patience, "full",
    )
    record(args, "JEPA -> full FT", metrics, latent)
    write_json(output / "config_jepa.json", vars(args))


if __name__ == "__main__":
    main()
