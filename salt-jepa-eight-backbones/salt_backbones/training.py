from __future__ import annotations

import json
import math
import random
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import SALTBackbone


@dataclass
class Metrics:
    mae: float
    mse: float
    rmse: float


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(datasets, batch_size: int, workers: int):
    return {
        split: DataLoader(
            dataset, batch_size=batch_size, shuffle=split == "train",
            num_workers=workers, pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@torch.no_grad()
def evaluate_forecast(model, loader, device) -> Metrics:
    model.eval()
    absolute = squared = 0.0
    count = 0
    for history, future in loader:
        error = model(history.to(device)) - future.to(device)
        absolute += float(error.abs().sum())
        squared += float(error.square().sum())
        count += error.numel()
    mse = squared / count
    return Metrics(absolute / count, mse, math.sqrt(mse))


def _unique_parameters(parameters):
    result, seen = [], set()
    for parameter in parameters:
        if id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return result


def fit_forecast(model: SALTBackbone, loaders, device, output: Path,
                 epochs: int, lr: float, encoder_lr_scale: float,
                 patience: int, strategy: str) -> Metrics:
    output.mkdir(parents=True, exist_ok=True)
    model.to(device)
    encoder = _unique_parameters(model.encoder_parameters())
    head = _unique_parameters(model.head_parameters())
    if strategy in ("full", "scratch", "supervised_source", "supervised_transfer"):
        groups = [
            {"params": encoder, "lr": lr * encoder_lr_scale, "name": "encoder"},
            {"params": head, "lr": lr, "name": "forecast_head"},
        ]
    else:
        raise ValueError(strategy)
    trainable = [p for group in groups for p in group["params"] if p.requires_grad]
    optimizer = torch.optim.AdamW(groups, weight_decay=3e-4)
    best, stale = math.inf, 0
    checkpoint = output / "best_forecast.pt"
    history_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for history, future in loaders["train"]:
            history, future = history.to(device), future.to(device)
            loss = nn.functional.l1_loss(model(history), future)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite forecast loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total += float(loss.detach()) * future.numel()
            count += future.numel()
        validation = evaluate_forecast(model, loaders["val"], device)
        history_log.append({"epoch": epoch, "train_mae": total / count, **asdict(validation)})
        print(
            f"{strategy} epoch {epoch:03d}: train_mae={total / count:.7f} "
            f"val_mae={validation.mae:.7f}", flush=True,
        )
        if validation.mae < best:
            best, stale = validation.mae, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test = evaluate_forecast(model, loaders["test"], device)
    write_json(output / "history.json", history_log)
    write_json(output / "metrics.json", asdict(test))
    return test


class SALTPredictor(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(student_dim), nn.Linear(student_dim, 2 * student_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * student_dim, teacher_dim), nn.LayerNorm(teacher_dim),
        )

    def forward(self, hidden):
        return self.network(hidden)


class Distiller(nn.Module):
    def __init__(self, teacher, student: SALTBackbone, dropout=0.1):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.predictor = SALTPredictor(student.latent_dim, teacher.model_dim, dropout)

    def forward(self, history, future):
        predicted = self.predictor(self.student.encode_history(history))
        with torch.no_grad():
            target = self.teacher(torch.cat((history, future), dim=1)).detach()
            target = self.student.align_teacher(target)
        if predicted.shape != target.shape:
            raise ValueError(f"Latent mismatch: Student {predicted.shape}, Teacher {target.shape}")
        return predicted, target


@torch.no_grad()
def evaluate_distillation(model: Distiller, loader, device, loss_name: str):
    model.eval()
    total, count = 0.0, 0
    for history, future in loader:
        prediction, target = model(history.to(device), future.to(device))
        if loss_name == "l2":
            error = (prediction - target).square()
        else:
            error = (prediction - target).abs()
        total += float(error.sum())
        count += error.numel()
    return total / count


def fit_distillation(student: SALTBackbone, teacher, loaders, device,
                     output: Path, epochs: int, lr: float, patience: int,
                     loss_name: str, dropout: float):
    output.mkdir(parents=True, exist_ok=True)
    model = Distiller(teacher, student, dropout).to(device)
    parameters = _unique_parameters(
        list(student.encoder_parameters()) + list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=3e-4)
    best, stale = math.inf, 0
    checkpoint = output / "best_student.pt"
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        model.teacher.eval()
        total, count = 0.0, 0
        for history, future in loaders["train"]:
            prediction, target = model(history.to(device), future.to(device))
            loss = (
                nn.functional.mse_loss(prediction, target)
                if loss_name == "l2"
                else nn.functional.l1_loss(prediction, target)
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite distillation loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += float(loss.detach()) * target.numel()
            count += target.numel()
        validation = evaluate_distillation(model, loaders["val"], device, loss_name)
        log.append({"epoch": epoch, "train": total / count, "validation": validation})
        print(
            f"distill epoch {epoch:03d}: train={total / count:.7f} "
            f"val={validation:.7f}", flush=True,
        )
        if validation < best:
            best, stale = validation, 0
            torch.save({
                "student": student.state_dict(),
                "predictor": model.predictor.state_dict(),
                "epoch": epoch,
                "val_latent": validation,
            }, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    saved = torch.load(checkpoint, map_location="cpu")
    student.load_state_dict(saved["student"])
    write_json(output / "history.json", log)
    write_json(output / "distillation.json", {
        "best_epoch": saved["epoch"], "val_latent": saved["val_latent"],
        "predictor_discarded_downstream": True,
    })
    return saved


class JEPADistiller(nn.Module):
    """Masked online encoder with a stop-gradient EMA target encoder."""

    def __init__(self, online: SALTBackbone, dropout=0.1):
        super().__init__()
        self.online = online
        self.target = copy.deepcopy(online)
        for parameter in self.target.parameters():
            parameter.requires_grad = False
        self.target.eval()
        self.predictor = SALTPredictor(online.latent_dim, online.latent_dim, dropout)
        encoder_ids = {id(parameter) for parameter in online.encoder_parameters()}
        self.encoder_names = [
            name for name, parameter in online.named_parameters()
            if id(parameter) in encoder_ids
        ]
        if not self.encoder_names:
            raise ValueError("JEPA online encoder has no trainable parameters")

    def forward(self, history, future):
        prediction = self.predictor(self.online.encode_history(history))
        with torch.no_grad():
            target = self.target.encode_sequence(torch.cat((history, future), dim=1)).detach()
        if prediction.shape != target.shape:
            raise ValueError(f"JEPA latent mismatch: {prediction.shape} versus {target.shape}")
        return prediction, target

    @torch.no_grad()
    def update_target(self, momentum: float):
        online_parameters = dict(self.online.named_parameters())
        target_parameters = dict(self.target.named_parameters())
        for name in self.encoder_names:
            target_parameters[name].mul_(momentum).add_(
                online_parameters[name], alpha=1.0 - momentum
            )
        # BatchNorm statistics are buffers rather than parameters. Copying them
        # prevents PatchTST's frozen target from using stale initialization.
        online_buffers = dict(self.online.named_buffers())
        target_buffers = dict(self.target.named_buffers())
        for name, value in online_buffers.items():
            if name in target_buffers and target_buffers[name].shape == value.shape:
                target_buffers[name].copy_(value)


@torch.no_grad()
def evaluate_jepa(model: JEPADistiller, loader, device, loss_name: str):
    model.eval()
    total, count = 0.0, 0
    for history, future in loader:
        prediction, target = model(history.to(device), future.to(device))
        error = (
            (prediction - target).square()
            if loss_name == "l2"
            else (prediction - target).abs()
        )
        total += float(error.sum())
        count += error.numel()
    return total / count


def fit_jepa(online: SALTBackbone, loaders, device, output: Path,
             epochs: int, lr: float, patience: int, loss_name: str,
             dropout: float, ema_momentum: float):
    output.mkdir(parents=True, exist_ok=True)
    model = JEPADistiller(online, dropout).to(device)
    parameters = _unique_parameters(
        list(online.encoder_parameters()) + list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=3e-4)
    best, stale = math.inf, 0
    checkpoint = output / "best_online.pt"
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        model.target.eval()
        total, count = 0.0, 0
        for history, future in loaders["train"]:
            prediction, target = model(history.to(device), future.to(device))
            loss = (
                nn.functional.mse_loss(prediction, target)
                if loss_name == "l2"
                else nn.functional.l1_loss(prediction, target)
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite JEPA loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            model.update_target(ema_momentum)
            total += float(loss.detach()) * target.numel()
            count += target.numel()
        validation = evaluate_jepa(model, loaders["val"], device, loss_name)
        log.append({"epoch": epoch, "train": total / count, "validation": validation})
        print(
            f"jepa epoch {epoch:03d}: train={total / count:.7f} "
            f"val={validation:.7f}", flush=True,
        )
        if validation < best:
            best, stale = validation, 0
            torch.save({
                "online": online.state_dict(),
                "target": model.target.state_dict(),
                "predictor": model.predictor.state_dict(),
                "epoch": epoch,
                "val_latent": validation,
                "ema_momentum": ema_momentum,
            }, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    saved = torch.load(checkpoint, map_location="cpu")
    online.load_state_dict(saved["online"])
    write_json(output / "history.json", log)
    write_json(output / "jepa.json", {
        "best_epoch": saved["epoch"],
        "val_latent": saved["val_latent"],
        "ema_momentum": ema_momentum,
        "downstream_encoder": "ema_target",
        "predictor_discarded_downstream": True,
    })
    return saved
