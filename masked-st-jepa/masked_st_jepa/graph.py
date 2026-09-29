from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def normalize_adjacency(adjacency: np.ndarray) -> torch.Tensor:
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adjacency.shape}")
    adjacency = np.maximum(adjacency, 0)
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency = adjacency + np.eye(len(adjacency), dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degree, 1e-12), -0.5)
    normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return torch.from_numpy(normalized.astype(np.float32))


def correlation_graph(values: np.ndarray, top_k: int) -> np.ndarray:
    """Build a train-only undirected top-k absolute-correlation graph."""
    correlation = np.nan_to_num(np.abs(np.corrcoef(values, rowvar=False)))
    np.fill_diagonal(correlation, 0)
    nodes = correlation.shape[0]
    top_k = min(max(1, top_k), max(1, nodes - 1))
    adjacency = np.zeros_like(correlation, dtype=np.float32)
    for node in range(nodes):
        neighbors = np.argpartition(correlation[node], -top_k)[-top_k:]
        adjacency[node, neighbors] = correlation[node, neighbors]
    return np.maximum(adjacency, adjacency.T)


def load_graph(
    mode: str,
    nodes: int,
    adjacency_path: str,
    training_values: np.ndarray,
    top_k: int,
) -> tuple[torch.Tensor, np.ndarray]:
    if mode == "file":
        if not adjacency_path:
            raise ValueError("--adjacency is required when --graph-mode file")
        raw = np.load(Path(adjacency_path))
    elif mode == "correlation":
        raw = correlation_graph(training_values, top_k)
    elif mode == "fully_connected":
        raw = np.ones((nodes, nodes), dtype=np.float32) - np.eye(nodes, dtype=np.float32)
    elif mode == "identity":
        raw = np.eye(nodes, dtype=np.float32)
    else:
        raise ValueError(f"Unknown graph mode: {mode}")
    if raw.shape != (nodes, nodes):
        raise ValueError(f"Graph has {raw.shape}, data has {nodes} nodes")
    return normalize_adjacency(raw), np.asarray(raw, dtype=np.float32)

