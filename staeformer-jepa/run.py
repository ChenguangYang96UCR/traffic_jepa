#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from staeformer_jepa.data import make_datasets
from staeformer_jepa.model import FutureJEPA, STAEformerEncoder, STAEformerForecast
from staeformer_jepa.training import MetricAccumulator, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Future-latent JEPA pretraining for STAEformer")
    parser.add_argument("--mode", choices=("compare", "pretrain"), default="compare")
    parser.add_argument("--data", required=True)
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--no-time-features", action="store_true")
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--pred-steps", type=int, default=3)
    parser.add_argument("--steps-per-day", type=int, default=288)
    parser.add_argument("--input-embedding-dim", type=int, default=24)
    parser.add_argument("--tod-embedding-dim", type=int, default=24)
    parser.add_argument("--dow-embedding-dim", type=int, default=24)
    parser.add_argument("--adaptive-embedding-dim", type=int, default=80)
    parser.add_argument("--feed-forward-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--predictor-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--forecast-loss", choices=("mae", "mse"), default="mae")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/fremont_12_to_3")
    return parser.parse_args()


def make_encoder(args, nodes: int, channels: int) -> STAEformerEncoder:
    time_features = not args.no_time_features
    return STAEformerEncoder(
        num_nodes=nodes,
        max_steps=args.input_steps,
        input_dim=channels,
        steps_per_day=args.steps_per_day,
        input_embedding_dim=args.input_embedding_dim,
        tod_embedding_dim=args.tod_embedding_dim if time_features else 0,
        dow_embedding_dim=args.dow_embedding_dim if time_features else 0,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        feed_forward_dim=args.feed_forward_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    )


def make_loaders(args, datasets):
    return {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )
        for split, dataset in datasets.items()
    }


@torch.no_grad()
def evaluate_jepa(model, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for history, future, _ in loader:
        prediction, target = model(history.to(device), future.to(device))
        total += float(torch.nn.functional.l1_loss(prediction, target, reduction="sum"))
        count += target.numel()
    return total / count


def pretrain_jepa(args, loaders, encoder, device, output: Path) -> Path:
    seed_everything(args.seed)
    model = FutureJEPA(
        encoder,
        args.input_steps,
        args.pred_steps,
        args.heads,
        args.predictor_layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.pretrain_lr,
        weight_decay=args.weight_decay,
    )
    best, stale = math.inf, 0
    checkpoint = output / "pretrain" / "best_jepa.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        model.target_encoder.eval()
        total, count = 0.0, 0
        for history, future, _ in loaders["train"]:
            history, future = history.to(device), future.to(device)
            prediction, target = model(history, future)
            loss = torch.nn.functional.l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.online_encoder.parameters()) + list(model.predictor.parameters()), 5.0
            )
            optimizer.step()
            model.update_target(args.ema_momentum)
            total += float(loss.detach()) * target.numel()
            count += target.numel()
        validation = evaluate_jepa(model, loaders["val"], device)
        print(
            f"pretrain {epoch:03d}/{args.pretrain_epochs}: "
            f"train_latent_l1={total / count:.6f} val_latent_l1={validation:.6f}"
        )
        if validation < best:
            best, stale = validation, 0
            torch.save(
                {
                    "online_encoder": model.online_encoder.state_dict(),
                    "target_encoder": model.target_encoder.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "epoch": epoch,
                    "val_latent_l1": validation,
                    "args": vars(args),
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping pretrain after epoch {epoch}")
                break
    return checkpoint


@torch.no_grad()
def evaluate_forecast(model, loader, device, pred_steps: int):
    model.eval()
    metrics = MetricAccumulator(pred_steps)
    for history, _, labels in loader:
        prediction = model(history.to(device))
        metrics.update(prediction, labels.to(device))
    return metrics.result()


def forecast_loss(prediction, target, name: str):
    if name == "mae":
        return torch.nn.functional.l1_loss(prediction, target)
    return torch.nn.functional.mse_loss(prediction, target)


def train_forecaster(args, loaders, encoder, strategy: str, device, output: Path):
    # Give all downstream variants the same batch ordering and dropout stream.
    seed_everything(args.seed + 100_000)
    model = STAEformerForecast(encoder, args.input_steps, args.pred_steps).to(device)
    if strategy == "frozen":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )
    best, stale = math.inf, 0
    checkpoint = output / strategy / "best_forecast.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        if strategy == "frozen":
            model.encoder.eval()
        total, count = 0.0, 0
        for history, _, labels in loaders["train"]:
            history, labels = history.to(device), labels.to(device)
            prediction = model(history)
            loss = forecast_loss(prediction, labels, args.forecast_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0
            )
            optimizer.step()
            total += float(loss.detach()) * labels.numel()
            count += labels.numel()
        validation = evaluate_forecast(model, loaders["val"], device, args.pred_steps)
        print(
            f"{strategy} {epoch:03d}/{args.finetune_epochs}: "
            f"train_{args.forecast_loss}={total / count:.6f} "
            f"val_mae={validation['mae']:.6f} val_rmse={validation['rmse']:.6f}"
        )
        if validation["mae"] < best:
            best, stale = validation["mae"], 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "validation": validation, "args": vars(args)},
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping {strategy} after epoch {epoch}")
                break
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    test = evaluate_forecast(model, loaders["test"], device, args.pred_steps)
    write_json(output / strategy / "test_metrics.json", test)
    print(f"{strategy} test: MAE={test['mae']:.7f} MSE={test['mse']:.7f} RMSE={test['rmse']:.7f}")
    return test


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    output = Path(args.output)
    datasets = make_datasets(args)
    loaders = make_loaders(args, datasets)
    sample_history, sample_future, sample_label = datasets["train"][0]
    nodes, channels = sample_history.shape[1], sample_history.shape[2]
    print(
        f"Device={device}; splits={{'train': {len(datasets['train'])}, 'val': {len(datasets['val'])}, "
        f"'test': {len(datasets['test'])}}}; history={tuple(sample_history.shape)}; "
        f"future={tuple(sample_future.shape)}; target={tuple(sample_label.shape)}"
    )

    initial_encoder = make_encoder(args, nodes, channels)
    pretrain_checkpoint = pretrain_jepa(args, loaders, initial_encoder, device, output)
    if args.mode == "pretrain":
        print(f"Saved JEPA checkpoint to {pretrain_checkpoint.resolve()}")
        return
    pretrained_state = torch.load(pretrain_checkpoint, map_location="cpu")["online_encoder"]
    results = {}

    seed_everything(args.seed)
    scratch_encoder = make_encoder(args, nodes, channels)
    results["staeformer_scratch"] = train_forecaster(
        args, loaders, scratch_encoder, "scratch", device, output
    )

    for strategy in ("frozen", "full"):
        encoder = make_encoder(args, nodes, channels)
        encoder.load_state_dict(pretrained_state)
        results[f"jepa_{strategy}"] = train_forecaster(
            args, loaders, encoder, strategy, device, output
        )

    lines = [f"{'Method':<32} {'MAE':>12} {'MSE':>12} {'RMSE':>12}", "-" * 71]
    labels = {
        "staeformer_scratch": "STAEformer from scratch",
        "jepa_frozen": "JEPA + frozen STAEformer",
        "jepa_full": "JEPA + full fine-tune",
    }
    for key in ("staeformer_scratch", "jepa_frozen", "jepa_full"):
        item = results[key]
        lines.append(
            f"{labels[key]:<32} {item['mae']:>12.7f} {item['mse']:>12.7f} {item['rmse']:>12.7f}"
        )
    summary = "\n".join(lines)
    print("\nFuture flow forecasting comparison\n" + summary)
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_json(output / "all_results.json", results)


if __name__ == "__main__":
    main()
