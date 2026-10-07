from __future__ import annotations

from pathlib import Path

import numpy as np


def load_adjacency(path: str | Path, nodes: int) -> np.ndarray:
    adj = np.asarray(np.load(path), dtype=np.float32)
    if adj.shape != (nodes, nodes):
        raise ValueError(f"Adjacency {path} is {adj.shape}; expected {(nodes, nodes)}")
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    return adj


def shortest_hops(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    hops = np.where(adj > 0, 1.0, 511.0).astype(np.float32)
    np.fill_diagonal(hops, 0.0)
    for k in range(n):
        hops = np.minimum(hops, hops[:, k, None] + hops[None, k, :])
    return np.minimum(hops, 511.0)


def laplacian_pe(adj: np.ndarray, dim: int) -> np.ndarray:
    a = np.maximum(adj, adj.T).astype(np.float64)
    a = a + np.eye(a.shape[0])
    degree = np.maximum(a.sum(1), 1e-8)
    norm = a / np.sqrt(degree[:, None] * degree[None, :])
    _, vecs = np.linalg.eigh(np.eye(a.shape[0]) - norm)
    available = vecs[:, 1 : 1 + min(dim, max(0, vecs.shape[1] - 1))]
    if available.shape[1] < dim:
        available = np.pad(available, ((0, 0), (0, dim - available.shape[1])))
    return available.astype(np.float32)


def mean_window_profiles(train_x: np.ndarray) -> np.ndarray:
    """Return one mean 12-step profile per node from released windows."""
    values = np.asarray(train_x)[..., 0]
    profiles = values.mean(axis=0).T
    profiles -= profiles.mean(axis=1, keepdims=True)
    scale = profiles.std(axis=1, keepdims=True)
    return profiles / np.maximum(scale, 1e-6)


def profile_distance(train_x: np.ndarray) -> np.ndarray:
    """Semantic distance proxy when the released windows lack absolute timestamps.

    Official PDFormer averages complete days before FastDTW. The released city
    object arrays are already windowed and cannot be safely reassembled into a
    continuous calendar. We therefore compare standardized mean window profiles.
    """
    profile = mean_window_profiles(train_x)
    return np.sqrt(np.square(profile[:, None, :] - profile[None, :, :]).sum(-1)).astype(np.float32)


def pattern_keys(train_x: np.ndarray, length: int = 3, clusters: int = 16) -> np.ndarray:
    from sklearn.cluster import KMeans

    x = np.asarray(train_x, dtype=np.float32)[..., 0]
    candidates = x[:, :length].transpose(0, 2, 1).reshape(-1, length)
    if len(candidates) > 50000:
        candidates = candidates[np.linspace(0, len(candidates) - 1, 50000).astype(int)]
    k = min(clusters, len(candidates))
    centers = KMeans(n_clusters=k, n_init=10, random_state=2026).fit(candidates).cluster_centers_
    if k < clusters:
        centers = np.concatenate([centers, np.repeat(centers[-1:], clusters - k, axis=0)])
    return centers[..., None].astype(np.float32)


def spectral_patch_indices(adj: np.ndarray, patch_count: int = 8):
    """Graph-aware deterministic partition for the official PatchSTG core."""
    n = adj.shape[0]
    patch_count = min(patch_count, n)
    patch_size = int(np.ceil(n / patch_count))
    a = np.maximum(adj, adj.T) + np.eye(n)
    degree = np.maximum(a.sum(1), 1e-8)
    lap = np.eye(n) - a / np.sqrt(degree[:, None] * degree[None, :])
    _, vecs = np.linalg.eigh(lap)
    order = np.argsort(vecs[:, 1] if n > 1 else np.arange(n))
    original, reordered, all_nodes = [], [], []
    for p, part in enumerate(np.array_split(order, patch_count)):
        members = part.tolist()
        original.extend(members)
        reordered.extend(range(p * patch_size, p * patch_size + len(members)))
        while len(members) < patch_size:
            score = a[members].sum(axis=0)
            score[members] = -np.inf
            candidate = int(np.argmax(score))
            if not np.isfinite(score[candidate]):
                candidate = members[-1]
            members.append(candidate)
        all_nodes.extend(members)
    return (
        np.asarray(original, dtype=np.int64),
        np.asarray(reordered, dtype=np.int64),
        np.asarray(all_nodes, dtype=np.int64),
        patch_size,
        patch_count,
    )
