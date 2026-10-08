#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from salt_backbones.data import load_city
from salt_backbones.graph import load_adjacency
from salt_backbones.shared_teacher import SharedTrafficTeacher
from salt_backbones.training import (
    SpatioTemporalBlockMasker,
    fit_shared_teacher,
    make_loaders,
    seed_everything,
)


def arguments():
    parser = argparse.ArgumentParser(
        description="Train one backbone-independent SALT Teacher on traffic tensors"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--adj", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--pred-steps", type=int, default=12)
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--graph-pe-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-blocks", type=int, default=8)
    parser.add_argument("--future-block-ratio", type=float, default=0.5)
    parser.add_argument("--mask-min-time", type=int, default=2)
    parser.add_argument("--mask-max-time", type=int, default=6)
    parser.add_argument("--mask-min-sensor-ratio", type=float, default=0.10)
    parser.add_argument("--mask-max-sensor-ratio", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def main():
    args = arguments()
    if args.input_steps != 12 or args.pred_steps not in (6, 9, 12):
        raise ValueError("Supported protocol: 12 history -> horizon in {6, 9, 12}")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    datasets = load_city(
        args.data, args.input_steps, args.pred_steps,
        args.traffic_feature, args.file_pattern,
    )
    adjacency = load_adjacency(args.adj, datasets["train"].num_nodes)
    model = SharedTrafficTeacher(
        adjacency=adjacency,
        input_steps=args.input_steps,
        pred_steps=args.pred_steps,
        latent_dim=args.latent_dim,
        layers=args.layers,
        heads=args.heads,
        ff_dim=args.ff_dim,
        graph_pe_dim=args.graph_pe_dim,
        dropout=args.dropout,
    )
    masker = SpatioTemporalBlockMasker(
        num_blocks=args.mask_blocks,
        min_time=args.mask_min_time,
        max_time=args.mask_max_time,
        min_sensor_ratio=args.mask_min_sensor_ratio,
        max_sensor_ratio=args.mask_max_sensor_ratio,
        future_start=args.input_steps,
        future_block_ratio=args.future_block_ratio,
    )
    saved = fit_shared_teacher(
        model,
        make_loaders(datasets, args.batch_size, args.workers),
        device,
        Path(args.output),
        args.epochs,
        args.lr,
        args.patience,
        masker,
        args.seed + 77_000,
        checkpoint_metadata={
            "source_data": str(Path(args.data).resolve()),
            "source_adj": str(Path(args.adj).resolve()),
            "mask_blocks": args.mask_blocks,
            "future_block_ratio": args.future_block_ratio,
            "mask_min_time": args.mask_min_time,
            "mask_max_time": args.mask_max_time,
            "mask_min_sensor_ratio": args.mask_min_sensor_ratio,
            "mask_max_sensor_ratio": args.mask_max_sensor_ratio,
        },
    )
    print(
        f"Saved shared Teacher: {Path(args.output, 'best_teacher.pt').resolve()} "
        f"(val reconstruction={saved['val_reconstruction']:.7f})"
    )


if __name__ == "__main__":
    main()
