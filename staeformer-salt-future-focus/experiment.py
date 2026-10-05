from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from staeformer_salt.data import make_datasets
from staeformer_salt.model import MaskedFutureForecast, STAEformerEncoder
from staeformer_salt.training import MetricAccumulator, seed_everything, write_json


def make_encoder(args, nodes: int, channels: int) -> STAEformerEncoder:
    return STAEformerEncoder(
        num_nodes=nodes,
        max_steps=args.input_steps + args.pred_steps,
        input_dim=channels,
        input_embedding_dim=args.input_embedding_dim,
        step_embedding_dim=args.step_embedding_dim,
        sensor_embedding_dim=args.sensor_embedding_dim,
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


def datasets_for(args, root: str):
    city_args = copy.copy(args)
    city_args.data = root
    return make_datasets(city_args)


def inspect_dataset(datasets, city: str):
    history, future, label = datasets["train"][0]
    print(
        f"{city}: splits={{'train': {len(datasets['train'])}, "
        f"'val': {len(datasets['val'])}, 'test': {len(datasets['test'])}}}; "
        f"history={tuple(history.shape)}; future={tuple(future.shape)}; "
        f"label={tuple(label.shape)}"
    )
    return history.shape[1], history.shape[2]


def load_cross_city_encoder(target_encoder, source_state: dict[str, torch.Tensor]) -> dict:
    """Transfer shared weights and keep a fresh target-city sensor table."""
    target_state = target_encoder.state_dict()
    loaded, city_specific, mismatched, unexpected = [], [], [], []
    for name, value in source_state.items():
        if name == "sensor_embedding":
            city_specific.append(name)
        elif name not in target_state:
            unexpected.append(name)
        elif target_state[name].shape != value.shape:
            mismatched.append({
                "name": name,
                "source": list(value.shape),
                "target": list(target_state[name].shape),
            })
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


def configure_downstream(model: MaskedFutureForecast, strategy: str, args):
    sensor = model.encoder.sensor_embedding
    if strategy == "frozen":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        groups = [{
            "params": model.output_projection.parameters(),
            "lr": args.finetune_lr,
            "name": "forecast_head",
        }]
        if not getattr(args, "freeze_city_embedding", False):
            sensor.requires_grad = True
            groups.insert(0, {
                "params": [sensor],
                "lr": args.finetune_lr,
                "name": "target_sensor_embedding",
            })
    elif strategy == "full":
        shared = [
            parameter for name, parameter in model.encoder.named_parameters()
            if name != "sensor_embedding"
        ]
        groups = [
            {
                "params": shared,
                "lr": args.finetune_lr * args.encoder_lr_scale,
                "name": "distilled_encoder",
            },
            {
                "params": [sensor],
                "lr": args.finetune_lr,
                "name": "target_sensor_embedding",
            },
            {
                "params": model.output_projection.parameters(),
                "lr": args.finetune_lr,
                "name": "forecast_head",
            },
        ]
    elif strategy == "scratch":
        groups = [{
            "params": model.parameters(),
            "lr": args.finetune_lr,
            "name": "scratch_model",
        }]
    else:
        raise ValueError(strategy)
    return groups


@torch.no_grad()
def evaluate_forecast(model, loader, device, pred_steps: int):
    model.eval()
    metrics = MetricAccumulator(pred_steps)
    for history, _, labels in loader:
        metrics.update(model(history.to(device)), labels.to(device))
    return metrics.result()


def train_downstream(
    args, loaders, encoder, strategy: str, device, output: Path,
    initial_head_state: dict[str, torch.Tensor] | None = None,
):
    seed_everything(args.seed + 100_000)
    model = MaskedFutureForecast(encoder, args.input_steps, args.pred_steps).to(device)
    if initial_head_state is not None:
        model.output_projection.load_state_dict(initial_head_state)
    optimizer = torch.optim.AdamW(
        configure_downstream(model, strategy, args), weight_decay=args.weight_decay
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
            if args.forecast_loss == "mae":
                loss = torch.nn.functional.l1_loss(prediction, labels)
            else:
                loss = torch.nn.functional.mse_loss(prediction, labels)
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
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "validation": validation, "args": vars(args),
            }, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping {strategy} after epoch {epoch}")
                break
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model"])
    test = evaluate_forecast(model, loaders["test"], device, args.pred_steps)
    test["selection_validation"] = saved["validation"]
    write_json(output / strategy / "test_metrics.json", test)
    print(f"{strategy} test: MAE={test['mae']:.7f} MSE={test['mse']:.7f} RMSE={test['rmse']:.7f}")
    return test
