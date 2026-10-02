from __future__ import annotations

import json
import math
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


class MetricAccumulator:
    def __init__(self, horizons: int):
        self.absolute = 0.0
        self.square = 0.0
        self.count = 0
        self.horizon_absolute = np.zeros(horizons, dtype=np.float64)
        self.horizon_square = np.zeros(horizons, dtype=np.float64)
        self.horizon_count = np.zeros(horizons, dtype=np.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        error = (prediction.detach() - target.detach()).float()
        self.absolute += float(error.abs().sum())
        self.square += float(error.square().sum())
        self.count += error.numel()
        for step in range(error.shape[1]):
            horizon = error[:, step]
            self.horizon_absolute[step] += float(horizon.abs().sum())
            self.horizon_square[step] += float(horizon.square().sum())
            self.horizon_count[step] += horizon.numel()

    def result(self) -> dict[str, object]:
        mse = self.square / self.count
        per_horizon = []
        for step in range(len(self.horizon_count)):
            step_mse = self.horizon_square[step] / self.horizon_count[step]
            per_horizon.append(
                {
                    "step": step + 1,
                    "mae": float(self.horizon_absolute[step] / self.horizon_count[step]),
                    "mse": float(step_mse),
                    "rmse": float(math.sqrt(step_mse)),
                }
            )
        return {
            "mae": float(self.absolute / self.count),
            "mse": float(mse),
            "rmse": float(math.sqrt(mse)),
            "per_horizon": per_horizon,
        }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
