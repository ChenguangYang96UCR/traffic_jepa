from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_losses(
    predicted_value: torch.Tensor,
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    selector = mask[..., None]
    predicted = predicted_value[selector]
    actual = truth[selector]
    latent_prediction = predicted_latent[mask]
    latent_target = target_latent[mask]
    value_loss = torch.nn.functional.smooth_l1_loss(predicted, actual)
    latent_loss = torch.nn.functional.l1_loss(latent_prediction, latent_target)
    error = predicted - actual
    return {
        "value": value_loss,
        "latent": latent_loss,
        "mae": error.abs().mean(),
        "mse": error.square().mean(),
    }


class AverageMetrics:
    def __init__(self):
        self.total: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, torch.Tensor], weight: int) -> None:
        self.count += weight
        for key, value in metrics.items():
            self.total[key] = self.total.get(key, 0.0) + float(value.detach()) * weight

    def result(self) -> dict[str, float]:
        result = {key: value / self.count for key, value in self.total.items()}
        if "mse" in result:
            result["rmse"] = result["mse"] ** 0.5
        return result


def save_checkpoint(path: Path, model, args, stats, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "normalization": stats,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

