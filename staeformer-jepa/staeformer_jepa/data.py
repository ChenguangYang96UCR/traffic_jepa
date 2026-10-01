from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _numpy_pickle_compatibility() -> None:
    """Allow NumPy 1.x to read trusted object arrays written by NumPy 2.x."""
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


class FremontForecastDataset(Dataset):
    """Past/future pairs from released Fremont incident windows.

    x_data and y_data are kept as separate chronological blocks. The model sees
    x_data only during downstream forecasting. y_data supplies the future flow
    labels and, during JEPA pretraining only, the stop-gradient target branch.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        input_steps: int = 12,
        pred_steps: int = 3,
        traffic_feature: int = 0,
        use_time_features: bool = True,
        pattern: str = "incident_{split}.npy",
    ):
        path = Path(root) / pattern.format(split=split)
        if not path.exists():
            raise FileNotFoundError(path)
        samples = np.load(path, allow_pickle=True)
        histories: list[np.ndarray] = []
        futures: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict) or not {"x_data", "y_data"} <= set(sample):
                raise ValueError(f"{path}: sample {index} must contain x_data and y_data")
            x = np.asarray(sample["x_data"], dtype=np.float32)
            y = np.asarray(sample["y_data"], dtype=np.float32)
            if x.ndim != 3 or y.ndim != 3 or x.shape[1] != y.shape[1]:
                raise ValueError(f"{path}: invalid x/y shapes {x.shape}/{y.shape}")
            if x.shape[0] < input_steps or y.shape[0] < pred_steps:
                raise ValueError(
                    f"{path}: need x>={input_steps}, y>={pred_steps}; got {x.shape}/{y.shape}"
                )
            if not 0 <= traffic_feature < min(x.shape[-1], y.shape[-1]):
                raise ValueError(f"traffic feature {traffic_feature} invalid for {x.shape}/{y.shape}")
            x = x[-input_steps:]
            y = y[:pred_steps]
            flow_x = x[..., traffic_feature : traffic_feature + 1]
            flow_y = y[..., traffic_feature : traffic_feature + 1]
            if use_time_features:
                if min(x.shape[-1], y.shape[-1]) < 3:
                    raise ValueError("Time features require channels 1=time-of-day and 2=day-of-week")
                self._validate_time_features(x, path, index)
                self._validate_time_features(y, path, index)
                model_x = np.concatenate((flow_x, x[..., 1:3]), axis=-1)
                model_y = np.concatenate((flow_y, y[..., 1:3]), axis=-1)
            else:
                model_x, model_y = flow_x, flow_y
            if not np.isfinite(model_x).all() or not np.isfinite(model_y).all():
                raise ValueError(f"{path}: nonfinite value in sample {index}")
            histories.append(model_x)
            futures.append(model_y)
            labels.append(flow_y)
        if not histories:
            raise ValueError(f"No samples in {path}")
        self.history = torch.from_numpy(np.stack(histories))
        self.future = torch.from_numpy(np.stack(futures))
        self.labels = torch.from_numpy(np.stack(labels))

    @staticmethod
    def _validate_time_features(data: np.ndarray, path: Path, index: int) -> None:
        time = data[:, 0, 1:3]
        if not np.allclose(data[..., 1:3], time[:, None, :], equal_nan=False):
            raise ValueError(f"{path}: time channels vary across sensors in sample {index}")
        tod, dow = time[:, 0], time[:, 1]
        if np.any(tod < 0) or np.any(tod >= 1.00001):
            raise ValueError(f"{path}: time-of-day must be normalized to [0,1] in sample {index}")
        if np.any(dow < 0) or np.any(dow > 6.00001):
            raise ValueError(f"{path}: day-of-week must be in [0,6] in sample {index}")

    def __len__(self) -> int:
        return self.history.shape[0]

    def __getitem__(self, index: int):
        return self.history[index], self.future[index], self.labels[index]


def make_datasets(args) -> dict[str, FremontForecastDataset]:
    return {
        split: FremontForecastDataset(
            args.data,
            split,
            input_steps=args.input_steps,
            pred_steps=args.pred_steps,
            traffic_feature=args.traffic_feature,
            use_time_features=not args.no_time_features,
            pattern=args.file_pattern,
        )
        for split in ("train", "val", "test")
    }

