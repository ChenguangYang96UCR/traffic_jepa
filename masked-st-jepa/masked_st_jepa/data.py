from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def _numpy_pickle_compatibility() -> None:
    """Allow NumPy 1.x to read trusted object arrays saved by NumPy 2.x."""
    try:
        importlib.import_module("numpy._core")
        return
    except ModuleNotFoundError:
        pass
    core = importlib.import_module("numpy.core")
    sys.modules.setdefault("numpy._core", core)
    for name in ("multiarray", "numeric", "_multiarray_umath", "umath"):
        sys.modules.setdefault(
            f"numpy._core.{name}", importlib.import_module(f"numpy.core.{name}")
        )


_numpy_pickle_compatibility()


class TrafficWindowDataset(Dataset):
    """A tensor-backed dataset whose samples are [time, node, 1]."""

    def __init__(self, windows: np.ndarray):
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim == 3:
            values = values[..., None]
        if values.ndim != 4 or values.shape[-1] != 1:
            raise ValueError(
                f"windows must be [sample,time,node] or [...,1], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Traffic windows contain NaN or infinity")
        self.values = torch.from_numpy(values)

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


def load_incident_windows(
    root: str | Path,
    split: str,
    feature: int = 0,
    pattern: str = "incident_{split}.npy",
) -> TrafficWindowDataset:
    """Read XTraffic object arrays and use x_data as a 12-snapshot graph sequence."""
    path = Path(root) / pattern.format(split=split)
    if not path.exists():
        raise FileNotFoundError(path)
    samples = np.load(path, allow_pickle=True)
    windows: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or "x_data" not in sample:
            raise ValueError(f"{path}: sample {index} has no x_data dictionary field")
        data = np.asarray(sample["x_data"])
        if data.ndim != 3 or not 0 <= feature < data.shape[-1]:
            raise ValueError(
                f"{path}: sample {index} has shape {data.shape}; feature={feature}"
            )
        windows.append(data[..., feature])
    if not windows:
        raise ValueError(f"No samples in {path}")
    return TrafficWindowDataset(np.stack(windows))


def _read_continuous(path: str | Path, npz_key: str = "data") -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        if npz_key not in archive:
            raise KeyError(f"{path} contains {archive.files}, not {npz_key!r}")
        values = archive[npz_key]
    elif suffix in {".csv", ".txt"}:
        import pandas as pd

        frame = pd.read_csv(path)
        # The Fremont builder writes date as the first column.
        if len(frame.columns) > 1 and not np.issubdtype(frame.iloc[:, 0].dtype, np.number):
            frame = frame.iloc[:, 1:]
        values = frame.to_numpy()
    else:
        raise ValueError(f"Unsupported continuous data format: {path}")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Continuous data must be [time,node], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(
            "Continuous data contains missing/nonfinite values; impute upstream to avoid leakage"
        )
    return values


def make_continuous_splits(
    path: str | Path,
    window: int = 12,
    stride: int = 1,
    ratios: Iterable[float] = (0.7, 0.1, 0.2),
    normalize: bool = True,
    npz_key: str = "data",
) -> tuple[dict[str, TrafficWindowDataset], dict[str, object]]:
    """Chronologically split first, then form windows without boundary leakage."""
    values = _read_continuous(path, npz_key=npz_key)
    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("ratios must contain three positive values")
    ratios = tuple(value / sum(ratios) for value in ratios)
    first = int(len(values) * ratios[0])
    second = first + int(len(values) * ratios[1])
    arrays = {
        "train": values[:first],
        "val": values[first:second],
        "test": values[second:],
    }
    mean = arrays["train"].mean(axis=0, keepdims=True)
    std = arrays["train"].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    datasets: dict[str, TrafficWindowDataset] = {}
    for split, array in arrays.items():
        if normalize:
            array = (array - mean) / std
        if len(array) < window:
            raise ValueError(f"{split} has {len(array)} rows, fewer than window={window}")
        starts = range(0, len(array) - window + 1, stride)
        datasets[split] = TrafficWindowDataset(
            np.stack([array[start : start + window] for start in starts])
        )
    stats: dict[str, object] = {
        "mean": mean.squeeze(0).tolist(),
        "std": std.squeeze(0).tolist(),
        "normalized": normalize,
        "source_rows": len(values),
    }
    return datasets, stats


def datasets_from_args(args) -> tuple[dict[str, TrafficWindowDataset], dict[str, object]]:
    if args.data_format == "incident":
        datasets = {
            split: load_incident_windows(
                args.data, split, feature=args.traffic_feature, pattern=args.file_pattern
            )
            for split in ("train", "val", "test")
        }
        return datasets, {"normalized": False, "source": "incident windows"}
    return make_continuous_splits(
        args.data,
        window=args.window,
        stride=args.stride,
        ratios=args.split_ratios,
        normalize=not args.no_normalize,
        npz_key=args.npz_key,
    )


def training_matrix(dataset: TrafficWindowDataset) -> np.ndarray:
    """Flatten sample and time for correlation-graph estimation."""
    return dataset.values[..., 0].numpy().reshape(-1, dataset.values.shape[2])

