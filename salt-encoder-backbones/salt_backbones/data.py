from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _numpy_pickle_compatibility() -> None:
    try:
        importlib.import_module("numpy._core")
    except ModuleNotFoundError:
        core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", core)
        for name in ("multiarray", "numeric", "_multiarray_umath", "umath"):
            sys.modules.setdefault(
                f"numpy._core.{name}", importlib.import_module(f"numpy.core.{name}")
            )


_numpy_pickle_compatibility()


class CityWindows(Dataset):
    """Released city windows, retaining a strict history/future boundary."""

    def __init__(self, root: str | Path, split: str, input_steps: int = 12,
                 pred_steps: int = 12, feature: int = 0,
                 pattern: str = "incident_{split}.npy"):
        self.path = Path(root) / pattern.format(split=split)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        samples = np.load(self.path, allow_pickle=True)
        histories, futures = [], []
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict) or not {"x_data", "y_data"} <= sample.keys():
                raise ValueError(f"{self.path}: sample {index} has no x_data/y_data")
            x = np.asarray(sample["x_data"], dtype=np.float32)
            y = np.asarray(sample["y_data"], dtype=np.float32)
            if x.ndim != 3 or y.ndim != 3 or x.shape[1] != y.shape[1]:
                raise ValueError(f"{self.path}: invalid shapes {x.shape}/{y.shape}")
            if x.shape[0] < input_steps or y.shape[0] < pred_steps:
                raise ValueError(f"{self.path}: requested {input_steps}->{pred_steps}, got {x.shape}/{y.shape}")
            if feature >= min(x.shape[-1], y.shape[-1]):
                raise ValueError(f"{self.path}: traffic feature {feature} is unavailable")
            history = x[-input_steps:, :, feature:feature + 1]
            future = y[:pred_steps, :, feature:feature + 1]
            if not np.isfinite(history).all() or not np.isfinite(future).all():
                raise ValueError(f"{self.path}: non-finite value at sample {index}")
            histories.append(history)
            futures.append(future)
        self.history = torch.from_numpy(np.stack(histories))
        self.future = torch.from_numpy(np.stack(futures))

    @property
    def num_nodes(self) -> int:
        return int(self.history.shape[2])

    def __len__(self) -> int:
        return int(self.history.shape[0])

    def __getitem__(self, index: int):
        return self.history[index], self.future[index]


def load_city(root, input_steps=12, pred_steps=12, feature=0,
              pattern="incident_{split}.npy"):
    datasets = {
        split: CityWindows(root, split, input_steps, pred_steps, feature, pattern)
        for split in ("train", "val", "test")
    }
    counts = {dataset.num_nodes for dataset in datasets.values()}
    if len(counts) != 1:
        raise ValueError(f"Node count differs across splits: {counts}")
    return datasets
