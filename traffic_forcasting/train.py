#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mstjepa.data import datasets_from_args, training_matrix
from mstjepa.graph import load_graph
from mstjepa.masking import SensorMasker
from mstjepa.model import MaskedSTJEPA
from mstjepa.train_utils import (
    AverageMetrics,
    masked_losses,
    save_checkpoint,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Masked spatio-temporal graph JEPA for traffic sensor imputation"
    )
    parser.add_argument("--data", required=True, help="Incident directory or continuous file")
    parser.add_argument("--data-format", choices=("incident", "continuous"), default="incident")
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--npz-key", default="data")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.7, 0.1, 0.2))
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--graph-mode", choices=("file", "correlation", "fully_connected", "identity"), default="file")
    parser.add_argument("--adjacency", default="")
    parser.add_argument("--graph-top-k", type=int, default=8)
    parser.add_argument("--mask-k", type=int, default=9)
    parser.add_argument("--mask-mode", choices=("random", "persistent", "spatial_block"), default="random")
    parser.add_argument("--causal", action="store_true", help="Prevent temporal attention from seeing future steps")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-strategy", choices=("full", "head"), default="full")
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/fremont_masked_stjepa")
    return parser.parse_args()


def make_loaders(args, datasets):
    return {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }


def run_epoch(model, loader, masker, device, stage, args, optimizer=None, seed=0):
    training = optimizer is not None
    model.train(training)
    # Target encoder must remain deterministic and gradient-free.
    model.target_encoder.eval()
    totals = AverageMetrics()
    generator = torch.Generator().manual_seed(seed)
    for values in loader:
        values = values.to(device, non_blocking=True)
        mask = masker(values.shape[0], values.shape[1], device, generator)
        with torch.set_grad_enabled(training):
            predicted_value, predicted_latent, target_latent = model(values, mask)
            metrics = masked_losses(
                predicted_value, predicted_latent, target_latent, values, mask
            )
            if stage == "pretrain":
                loss = metrics["latent"]
            else:
                loss = metrics["value"] + args.jepa_weight * metrics["latent"]
            metrics["loss"] = loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0
                )
                optimizer.step()
                model.update_target(args.ema_momentum)
        totals.update(metrics, int(mask.sum()))
    return totals.result()


def fit_stage(model, loaders, masker, device, stage, epochs, args, output):
    if epochs <= 0:
        return None
    if stage == "finetune":
        model.set_finetune_strategy(args.finetune_strategy)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    best = math.inf
    stale = 0
    checkpoint = output / f"best_{stage}.pt"
    for epoch in range(1, epochs + 1):
        train = run_epoch(
            model, loaders["train"], masker, device, stage, args, optimizer,
            seed=args.seed + epoch,
        )
        validation = run_epoch(
            model, loaders["val"], masker, device, stage, args,
            seed=args.seed + 10_000,
        )
        key = validation["latent"] if stage == "pretrain" else validation["mae"]
        print(
            f"{stage} {epoch:03d}/{epochs}: "
            f"train_loss={train['loss']:.6f} val_loss={validation['loss']:.6f} "
            f"val_mae={validation['mae']:.6f} val_rmse={validation['rmse']:.6f}"
        )
        if key < best:
            best = key
            stale = 0
            save_checkpoint(checkpoint, model, args, {}, epoch, validation)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping {stage} after {epoch} epochs")
                break
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    return checkpoint


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    datasets, stats = datasets_from_args(args)
    sample = datasets["train"][0]
    steps, nodes, features = sample.shape
    if args.data_format == "incident" and steps != args.window:
        print(f"Using incident window length {steps}; --window={args.window} is ignored")
    adjacency, raw_adjacency = load_graph(
        args.graph_mode,
        nodes,
        args.adjacency,
        training_matrix(datasets["train"]),
        args.graph_top_k,
    )
    model = MaskedSTJEPA(
        nodes,
        steps,
        adjacency,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        causal=args.causal,
    ).to(device)
    masker = SensorMasker(nodes, args.mask_k, args.mask_mode, raw_adjacency)
    loaders = make_loaders(args, datasets)
    print(
        f"Device={device}; data={[ (k, len(v)) for k,v in datasets.items() ]}; "
        f"sample={tuple(sample.shape)}; mask={args.mask_mode}:{args.mask_k}"
    )

    pretrain_checkpoint = fit_stage(
        model, loaders, masker, device, "pretrain", args.pretrain_epochs, args, output
    )
    finetune_checkpoint = fit_stage(
        model, loaders, masker, device, "finetune", args.finetune_epochs, args, output
    )
    test = run_epoch(
        model, loaders["test"], masker, device, "finetune", args,
        seed=args.seed + 20_000,
    )
    result = {
        "task": "causal sensor forecasting" if args.causal else "sensor-time imputation",
        "shape": {"steps": steps, "nodes": nodes, "features": features},
        "mask": {"mode": args.mask_mode, "k_per_step": args.mask_k},
        "pretrain_checkpoint": str(pretrain_checkpoint) if pretrain_checkpoint else None,
        "finetune_checkpoint": str(finetune_checkpoint) if finetune_checkpoint else None,
        "test_masked_positions": test,
        "normalization": stats,
    }
    write_json(output / "test_metrics.json", result)
    print("Test masked positions:", test)
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()

