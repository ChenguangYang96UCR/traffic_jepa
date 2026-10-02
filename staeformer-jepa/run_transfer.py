#!/usr/bin/env python3
"""Oakland JEPA pretraining -> Fremont STAEformer forecasting transfer."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch

from run import (
    evaluate_forecast,
    forecast_loss,
    make_encoder,
    make_loaders,
    pretrain_jepa,
)
from staeformer_jepa.data import make_datasets
from staeformer_jepa.model import MaskedFutureForecast
from staeformer_jepa.training import seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a traffic-only JEPA encoder on one city and transfer it to another"
    )
    parser.add_argument("--mode", choices=("compare", "pretrain", "finetune"), default="compare")
    parser.add_argument("--source-data", required=True, help="Oakland split directory")
    parser.add_argument("--target-data", required=True, help="Fremont split directory")
    parser.add_argument("--pretrained-checkpoint", default="")
    parser.add_argument("--transfer-branch", choices=("target", "online"), default="target")
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--pred-steps", type=int, default=3)
    parser.add_argument(
        "--masked-steps", type=int, default=3,
        help="Number of complete masked steps; must equal pred-steps",
    )
    parser.add_argument(
        "--pretrain-mask-mode",
        choices=("future", "random"),
        default="future",
        help="future is leakage-free forecasting; random is retained only as an ablation",
    )
    parser.add_argument("--input-embedding-dim", type=int, default=24)
    parser.add_argument("--step-embedding-dim", type=int, default=24)
    parser.add_argument("--sensor-embedding-dim", type=int, default=80)
    parser.add_argument("--feed-forward-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--predictor-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument(
        "--encoder-lr-scale", type=float, default=0.1,
        help="LR multiplier for transferred shared encoder weights during full fine-tuning",
    )
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--forecast-loss", choices=("mae", "mse"), default="mae")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/oakland_to_fremont")
    return parser.parse_args()


def datasets_for(args: argparse.Namespace, root: str):
    city_args = copy.copy(args)
    city_args.data = root
    return make_datasets(city_args)


def inspect_dataset(datasets, city: str):
    history, future, label = datasets["train"][0]
    print(
        f"{city}: splits={{'train': {len(datasets['train'])}, 'val': {len(datasets['val'])}, "
        f"'test': {len(datasets['test'])}}}; history={tuple(history.shape)}; "
        f"future={tuple(future.shape)}; label={tuple(label.shape)}"
    )
    return history.shape[1], history.shape[2]


def load_cross_city_encoder(target_encoder, source_state: dict[str, torch.Tensor]) -> dict:
    """Load transferable weights while retaining a fresh target sensor table."""
    target_state = target_encoder.state_dict()
    loaded, city_specific, mismatched, unexpected = [], [], [], []
    for name, value in source_state.items():
        if name == "sensor_embedding":
            city_specific.append(name)
        elif name not in target_state:
            unexpected.append(name)
        elif target_state[name].shape != value.shape:
            mismatched.append(
                {"name": name, "source": list(value.shape), "target": list(target_state[name].shape)}
            )
        else:
            target_state[name] = value
            loaded.append(name)
    target_encoder.load_state_dict(target_state)
    if not loaded:
        raise ValueError("No compatible encoder weights were loaded")
    return {
        "loaded": loaded,
        "reinitialized_city_specific": city_specific,
        "shape_mismatch": mismatched,
        "unexpected": unexpected,
    }


def configure_transfer(model: MaskedFutureForecast, strategy: str, args):
    sensor = model.encoder.sensor_embedding
    if strategy == "frozen":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        sensor.requires_grad = True
        groups = [
            {"params": [sensor], "lr": args.finetune_lr, "name": "fremont_sensor_embedding"},
            {"params": model.output_projection.parameters(), "lr": args.finetune_lr, "name": "forecast_head"},
        ]
    elif strategy == "full":
        shared = [
            parameter
            for name, parameter in model.encoder.named_parameters()
            if name != "sensor_embedding"
        ]
        groups = [
            {
                "params": shared,
                "lr": args.finetune_lr * args.encoder_lr_scale,
                "name": "transferred_encoder",
            },
            {"params": [sensor], "lr": args.finetune_lr, "name": "fremont_sensor_embedding"},
            {"params": model.output_projection.parameters(), "lr": args.finetune_lr, "name": "forecast_head"},
        ]
    elif strategy == "scratch":
        groups = [{"params": model.parameters(), "lr": args.finetune_lr, "name": "scratch_model"}]
    else:
        raise ValueError(strategy)
    return groups


def train_target(args, loaders, encoder, strategy: str, device, output: Path):
    seed_everything(args.seed + 100_000)
    model = MaskedFutureForecast(encoder, args.input_steps, args.pred_steps).to(device)
    groups = configure_transfer(model, strategy, args)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    best, stale = math.inf, 0
    checkpoint = output / strategy / "best_forecast.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        if strategy == "frozen":
            # Evaluation mode makes the frozen dropout path deterministic;
            # gradients still reach the new sensor embedding and forecast head.
            model.encoder.eval()
        total, count = 0.0, 0
        for history, _, labels in loaders["train"]:
            history, labels = history.to(device), labels.to(device)
            prediction = model(history)
            loss = forecast_loss(prediction, labels, args.forecast_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
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

    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model"])
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

    source_datasets = datasets_for(args, args.source_data)
    source_nodes, source_channels = inspect_dataset(source_datasets, "source/Oakland")
    source_loaders = make_loaders(args, source_datasets)

    if args.mode in ("compare", "pretrain") and not args.pretrained_checkpoint:
        source_encoder = make_encoder(args, source_nodes, source_channels)
        checkpoint = pretrain_jepa(args, source_loaders, source_encoder, device, output / "source")
    elif args.pretrained_checkpoint:
        checkpoint = Path(args.pretrained_checkpoint)
    else:
        raise ValueError("--mode finetune requires --pretrained-checkpoint")

    if args.mode == "pretrain":
        print(f"Saved source JEPA checkpoint to {checkpoint.resolve()}")
        return

    target_datasets = datasets_for(args, args.target_data)
    target_nodes, target_channels = inspect_dataset(target_datasets, "target/Fremont")
    if source_channels != target_channels:
        raise ValueError(
            f"Source/target traffic channel count differs: {source_channels}/{target_channels}"
        )
    target_loaders = make_loaders(args, target_datasets)
    saved = torch.load(checkpoint, map_location="cpu")
    state_key = f"{args.transfer_branch}_encoder"
    source_state = saved[state_key]

    results = {}
    seed_everything(args.seed)
    scratch = make_encoder(args, target_nodes, target_channels)
    results["staeformer_scratch"] = train_target(
        args, target_loaders, scratch, "scratch", device, output
    )

    transfer_report = None
    for strategy in ("frozen", "full"):
        target_encoder = make_encoder(args, target_nodes, target_channels)
        report = load_cross_city_encoder(target_encoder, source_state)
        transfer_report = report
        results[f"jepa_{args.transfer_branch}_{strategy}"] = train_target(
            args, target_loaders, target_encoder, strategy, device, output
        )

    report_payload = {
        "source_nodes": source_nodes,
        "target_nodes": target_nodes,
        "transferred_branch": args.transfer_branch,
        **transfer_report,
    }
    write_json(output / "transfer_report.json", report_payload)

    order = [
        "staeformer_scratch",
        f"jepa_{args.transfer_branch}_frozen",
        f"jepa_{args.transfer_branch}_full",
    ]
    labels = {
        "staeformer_scratch": "Fremont masked-future scratch",
        f"jepa_{args.transfer_branch}_frozen": f"Oakland {args.pretrain_mask_mode}-mask {args.transfer_branch} -> frozen",
        f"jepa_{args.transfer_branch}_full": f"Oakland {args.pretrain_mask_mode}-mask {args.transfer_branch} -> full FT",
    }
    lines = [f"{'Method':<42} {'MAE':>12} {'MSE':>12} {'RMSE':>12}", "-" * 81]
    for key in order:
        item = results[key]
        lines.append(
            f"{labels[key]:<42} {item['mae']:>12.7f} {item['mse']:>12.7f} {item['rmse']:>12.7f}"
        )
    summary = "\n".join(lines)
    print("\nOakland -> Fremont transfer comparison\n" + summary)
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_json(output / "all_results.json", results)


if __name__ == "__main__":
    main()
