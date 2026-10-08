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
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .models import SALTBackbone
from .shared_teacher import SharedTrafficTeacher


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


class SpatioTemporalBlockMasker:
    """Sample the same future-biased block masks used by the SALT sweep."""

    def __init__(self, num_blocks=8, min_time=2, max_time=6,
                 min_sensor_ratio=0.10, max_sensor_ratio=0.30,
                 future_start=12, future_block_ratio=0.5):
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if not 0 < min_time <= max_time:
            raise ValueError("time block sizes must satisfy 0 < min <= max")
        if not 0 < min_sensor_ratio <= max_sensor_ratio <= 1:
            raise ValueError("sensor ratios must satisfy 0 < min <= max <= 1")
        if not 0 <= future_block_ratio <= 1:
            raise ValueError("future_block_ratio must be in [0, 1]")
        self.num_blocks = num_blocks
        self.min_time = min_time
        self.max_time = max_time
        self.min_sensor_ratio = min_sensor_ratio
        self.max_sensor_ratio = max_sensor_ratio
        self.future_start = future_start
        self.future_block_ratio = future_block_ratio

    def _regions(self, steps):
        if not 0 < self.future_start < steps:
            raise ValueError(
                f"future_start={self.future_start} must be inside a {steps}-step sequence"
            )
        future_blocks = int(self.num_blocks * self.future_block_ratio + 0.5)
        future_blocks = min(self.num_blocks, max(0, future_blocks))
        return (
            [(self.future_start, steps)] * future_blocks
            + [(0, self.future_start)] * (self.num_blocks - future_blocks)
        )

    def __call__(self, batch, steps, nodes, device, generator=None):
        mask = torch.zeros(batch, steps, nodes, dtype=torch.bool, device=device)
        min_nodes = max(1, round(nodes * self.min_sensor_ratio))
        max_nodes = max(min_nodes, round(nodes * self.max_sensor_ratio))
        for sample in range(batch):
            for region_start, region_end in self._regions(steps):
                region_steps = region_end - region_start
                max_time = min(self.max_time, region_steps)
                min_time = min(self.min_time, max_time)
                length = int(torch.randint(
                    min_time, max_time + 1, (1,), device=device, generator=generator,
                ).item())
                local_start = int(torch.randint(
                    0, region_steps - length + 1, (1,),
                    device=device, generator=generator,
                ).item())
                count = int(torch.randint(
                    min_nodes, max_nodes + 1, (1,),
                    device=device, generator=generator,
                ).item())
                sensors = torch.randperm(nodes, device=device, generator=generator)[:count]
                start = region_start + local_start
                mask[sample, start:start + length, sensors] = True
        return mask


class BackboneReconstructionTeacher(nn.Module):
    """A reconstruction Teacher whose encoder is the selected backbone itself."""

    def __init__(self, backbone: SALTBackbone):
        super().__init__()
        self.backbone = backbone
        self.decoder = nn.Linear(backbone.latent_dim, 1)

    @staticmethod
    def _restore_time_resolution(hidden, steps):
        if hidden.shape[1] == steps:
            return hidden
        batch, latent_steps, nodes, dim = hidden.shape
        values = hidden.permute(0, 2, 3, 1).reshape(batch * nodes, dim, latent_steps)
        values = F.interpolate(values, size=steps, mode="linear", align_corners=False)
        return values.reshape(batch, nodes, dim, steps).permute(0, 3, 1, 2)

    def forward(self, sequence, mask):
        masked = sequence.clone()
        token = self.backbone.future_token.to(sequence)
        masked[mask.unsqueeze(-1).expand_as(masked)] = token
        hidden = self.backbone.encode_sequence(masked)
        hidden = self._restore_time_resolution(hidden, sequence.shape[1])
        return self.decoder(hidden)


@torch.no_grad()
def evaluate_teacher(model, loader, masker, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = count = masked_count = future_masked = 0
    for history, future in loader:
        sequence = torch.cat((history, future), dim=1).to(device)
        mask = masker(*sequence.shape[:3], device, generator)
        error = F.smooth_l1_loss(model(sequence, mask), sequence, reduction="none")
        selected = error[mask.unsqueeze(-1).expand_as(error)]
        total += float(selected.sum())
        count += selected.numel()
        masked_count += int(mask.sum())
        future_masked += int(mask[:, masker.future_start:].sum())
    return {
        "reconstruction": total / count,
        "mask_fraction_future": future_masked / masked_count,
    }


def fit_masked_teacher(backbone: SALTBackbone, loaders, device, output: Path,
                       epochs: int, lr: float, patience: int, masker,
                       validation_seed: int, checkpoint_metadata=None):
    output.mkdir(parents=True, exist_ok=True)
    model = BackboneReconstructionTeacher(backbone).to(device)
    parameters = _unique_parameters(
        list(backbone.encoder_parameters()) + list(model.decoder.parameters())
    )
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=3e-4)
    checkpoint = output / "best_teacher.pt"
    best, stale = math.inf, 0
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = count = mask_count = future_mask_count = value_count = 0
        for history, future in loaders["train"]:
            sequence = torch.cat((history, future), dim=1).to(device)
            mask = masker(*sequence.shape[:3], device)
            prediction = model(sequence, mask)
            selected = mask.unsqueeze(-1).expand_as(prediction)
            loss = F.smooth_l1_loss(prediction[selected], sequence[selected])
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite Teacher loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += float(loss.detach()) * int(selected.sum())
            count += int(selected.sum())
            mask_count += int(mask.sum())
            future_mask_count += int(mask[:, masker.future_start:].sum())
            value_count += mask.numel()
        validation = evaluate_teacher(
            model, loaders["val"], masker, device, validation_seed,
        )
        row = {
            "epoch": epoch, "train_reconstruction": total / count,
            "val_reconstruction": validation["reconstruction"],
            "mask_density": mask_count / value_count,
            "train_future_mask_fraction": future_mask_count / mask_count,
            "val_future_mask_fraction": validation["mask_fraction_future"],
        }
        log.append(row)
        print(
            f"teacher epoch {epoch:03d}: train={total / count:.7f} "
            f"val={validation['reconstruction']:.7f} "
            f"mask_density={mask_count / value_count:.4f} "
            f"future_mask_fraction={validation['mask_fraction_future']:.4f}",
            flush=True,
        )
        if validation["reconstruction"] < best:
            best, stale = validation["reconstruction"], 0
            torch.save({
                "backbone": backbone.state_dict(),
                "decoder": model.decoder.state_dict(),
                "epoch": epoch,
                "val_reconstruction": validation["reconstruction"],
                "val_future_mask_fraction": validation["mask_fraction_future"],
                "metadata": dict(checkpoint_metadata or {}),
            }, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    saved = torch.load(checkpoint, map_location="cpu")
    backbone.load_state_dict(saved["backbone"])
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    write_json(output / "history.json", log)
    write_json(output / "teacher.json", {
        "best_epoch": saved["epoch"],
        "val_reconstruction": saved["val_reconstruction"],
        "val_future_mask_fraction": saved["val_future_mask_fraction"],
        "encoder": type(backbone).__name__,
        "metadata": saved.get("metadata", {}),
    })
    return saved


@torch.no_grad()
def evaluate_shared_teacher(model, loader, masker, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = count = masked_count = future_masked = 0
    for history, future in loader:
        sequence = torch.cat((history, future), dim=1).to(device)
        mask = masker(*sequence.shape[:3], device, generator)
        prediction = model.reconstruct(sequence, mask)
        selected = mask.unsqueeze(-1).expand_as(prediction)
        error = F.smooth_l1_loss(prediction[selected], sequence[selected], reduction="sum")
        total += float(error)
        count += int(selected.sum())
        masked_count += int(mask.sum())
        future_masked += int(mask[:, masker.future_start:].sum())
    return {
        "reconstruction": total / count,
        "mask_fraction_future": future_masked / masked_count,
    }


def fit_shared_teacher(model: SharedTrafficTeacher, loaders, device, output: Path,
                       epochs: int, lr: float, patience: int, masker,
                       validation_seed: int, checkpoint_metadata=None):
    """Stage 1: train one reusable Teacher, independent of Student backbone."""
    output.mkdir(parents=True, exist_ok=True)
    model.to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=3e-4)
    checkpoint = output / "best_teacher.pt"
    best, stale = math.inf, 0
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = count = mask_count = future_mask_count = value_count = 0
        for history, future in loaders["train"]:
            sequence = torch.cat((history, future), dim=1).to(device)
            mask = masker(*sequence.shape[:3], device)
            prediction = model.reconstruct(sequence, mask)
            selected = mask.unsqueeze(-1).expand_as(prediction)
            loss = F.smooth_l1_loss(prediction[selected], sequence[selected])
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite shared Teacher loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += float(loss.detach()) * int(selected.sum())
            count += int(selected.sum())
            mask_count += int(mask.sum())
            future_mask_count += int(mask[:, masker.future_start:].sum())
            value_count += mask.numel()
        validation = evaluate_shared_teacher(
            model, loaders["val"], masker, device, validation_seed,
        )
        row = {
            "epoch": epoch,
            "train_reconstruction": total / count,
            "val_reconstruction": validation["reconstruction"],
            "mask_density": mask_count / value_count,
            "train_future_mask_fraction": future_mask_count / mask_count,
            "val_future_mask_fraction": validation["mask_fraction_future"],
        }
        log.append(row)
        print(
            f"shared teacher epoch {epoch:03d}: train={total / count:.7f} "
            f"val={validation['reconstruction']:.7f} "
            f"mask_density={mask_count / value_count:.4f} "
            f"future_mask_fraction={validation['mask_fraction_future']:.4f}",
            flush=True,
        )
        if validation["reconstruction"] < best:
            best, stale = validation["reconstruction"], 0
            metadata = {
                **model.checkpoint_metadata(),
                "dropout": model.dropout,
                **dict(checkpoint_metadata or {}),
            }
            torch.save({
                "teacher": model.state_dict(),
                "epoch": epoch,
                "val_reconstruction": validation["reconstruction"],
                "val_future_mask_fraction": validation["mask_fraction_future"],
                "metadata": metadata,
            }, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    saved = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(saved["teacher"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    write_json(output / "history.json", log)
    write_json(output / "teacher.json", {
        "best_epoch": saved["epoch"],
        "val_reconstruction": saved["val_reconstruction"],
        "val_future_mask_fraction": saved["val_future_mask_fraction"],
        "metadata": saved["metadata"],
    })
    return saved


class Distiller(nn.Module):
    def __init__(self, teacher: nn.Module, student: SALTBackbone, dropout=0.1):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.predictor = SALTPredictor(student.latent_dim, teacher.latent_dim, dropout)

    def forward(self, history, future):
        predicted = self.predictor(self.student.encode_history(history))
        with torch.no_grad():
            target = self.teacher.encode_sequence(torch.cat((history, future), dim=1)).detach()
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
