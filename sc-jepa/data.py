"""Traffic-only paired windows, with the same feature/split contract as TopoJEPA."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficWindows(Dataset):
    def __init__(self, root, split, seq_len=12, pred_len=12, feature=0,
                 nodes=93, pattern='incident_{flag}.npy'):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Expected train, val or test')
        path = Path(root) / pattern.format(flag=split)
        # Only load trusted local files: these existing datasets contain pickles.
        samples = np.load(path, allow_pickle=True)
        if not len(samples):
            raise ValueError(f'Empty dataset: {path}')
        xs, ys = [], []
        for i, sample in enumerate(samples):
            if not isinstance(sample, dict) or not {'x_data', 'y_data'} <= sample.keys():
                raise ValueError(f'{path}: sample {i} needs x_data/y_data')
            x, y = np.asarray(sample['x_data']), np.asarray(sample['y_data'])
            if (x.ndim != 3 or y.ndim != 3 or x.shape[0] != seq_len or
                    y.shape[0] < pred_len or x.shape[1] != nodes or y.shape[1] != nodes or
                    not 0 <= feature < min(x.shape[2], y.shape[2])):
                raise ValueError(f'{path}: invalid sample {i}: x={x.shape}, y={y.shape}')
            x = np.asarray(x[:, :, feature], dtype=np.float32)
            y = np.asarray(y[:pred_len, :, feature], dtype=np.float32)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'{path}: nonfinite traffic in sample {i}')
            xs.append(x)
            ys.append(y)
        self.x = torch.from_numpy(np.stack(xs))
        self.y = torch.from_numpy(np.stack(ys))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
