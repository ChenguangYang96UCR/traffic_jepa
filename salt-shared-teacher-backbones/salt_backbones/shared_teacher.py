from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _normalized_adjacency(adjacency: np.ndarray) -> torch.Tensor:
    matrix = np.asarray(adjacency, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Adjacency must be square, got {matrix.shape}")
    matrix = np.maximum(matrix, 0.0)
    matrix = matrix + np.eye(matrix.shape[0], dtype=np.float32)
    degree = matrix.sum(axis=1)
    inverse = np.power(np.maximum(degree, 1e-8), -0.5)
    return torch.from_numpy(inverse[:, None] * matrix * inverse[None, :])


def _laplacian_features(adjacency: np.ndarray, width: int) -> torch.Tensor:
    """Deterministic node structure; no learned city-specific lookup table."""
    symmetric = np.maximum(
        np.asarray(adjacency, dtype=np.float32),
        np.asarray(adjacency, dtype=np.float32).T,
    )
    normalized = _normalized_adjacency(symmetric).numpy()
    laplacian = np.eye(normalized.shape[0], dtype=np.float32) - normalized
    _, vectors = np.linalg.eigh(laplacian)
    usable = vectors[:, 1:1 + min(width, max(0, vectors.shape[1] - 1))]
    # Eigenvectors have arbitrary signs. Fix them for reproducible checkpoints.
    for column in range(usable.shape[1]):
        pivot = int(np.argmax(np.abs(usable[:, column])))
        if usable[pivot, column] < 0:
            usable[:, column] *= -1
    if usable.shape[1] < width:
        usable = np.pad(usable, ((0, 0), (0, width - usable.shape[1])))
    return torch.from_numpy(usable.astype(np.float32))


class SharedTrafficBlock(nn.Module):
    """Alternating temporal attention, spatial attention and graph diffusion."""

    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        layer = dict(
            d_model=dim, nhead=heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoderLayer(**layer)
        self.spatial = nn.TransformerEncoderLayer(**layer)
        self.graph_gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
        self.graph_projection = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, hidden: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, dim = hidden.shape
        temporal = self.temporal(hidden.transpose(1, 2).reshape(batch * nodes, steps, dim))
        hidden = temporal.reshape(batch, nodes, steps, dim).transpose(1, 2)
        spatial = self.spatial(hidden.reshape(batch * steps, nodes, dim))
        hidden = spatial.reshape(batch, steps, nodes, dim)
        diffused = torch.einsum("nm,btmd->btnd", graph, hidden)
        diffused = self.graph_projection(diffused)
        gate = self.graph_gate(torch.cat((hidden, diffused), dim=-1))
        return self.output_norm(hidden + gate * diffused)


class SharedTrafficTeacher(nn.Module):
    """Backbone-independent Teacher over the complete traffic space-time tensor.

    Output always has shape [batch, history+future, sensors, latent_dim], so one
    checkpoint can supervise every Student adapter. City structure is injected
    through deterministic graph buffers rather than learned node IDs.
    """

    architecture = "shared_spatiotemporal_graph_transformer_v1"

    def __init__(
        self,
        adjacency: np.ndarray,
        input_steps: int = 12,
        pred_steps: int = 12,
        input_dim: int = 1,
        latent_dim: int = 128,
        layers: int = 3,
        heads: int = 8,
        ff_dim: int = 512,
        graph_pe_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by heads")
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.total_steps = input_steps + pred_steps
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers = layers
        self.heads = heads
        self.ff_dim = ff_dim
        self.graph_pe_dim = graph_pe_dim
        self.dropout = dropout
        graph = _normalized_adjacency(adjacency)
        self.register_buffer("normalized_adjacency", graph)
        self.register_buffer("graph_pe", _laplacian_features(adjacency, graph_pe_dim))
        degree = graph.sum(dim=1, keepdim=True)
        self.register_buffer("degree_feature", degree)

        # raw value, temporal derivative, graph-neighbour mean, and mask flag
        self.traffic_projection = nn.Linear(3 * input_dim + 1, latent_dim)
        self.graph_projection = nn.Linear(graph_pe_dim + 1, latent_dim)
        self.time_embedding = nn.Parameter(torch.empty(self.total_steps, latent_dim))
        self.segment_embedding = nn.Embedding(2, latent_dim)
        self.mask_token = nn.Parameter(torch.empty(latent_dim))
        self.input_norm = nn.LayerNorm(latent_dim)
        self.blocks = nn.ModuleList([
            SharedTrafficBlock(latent_dim, heads, ff_dim, dropout)
            for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(latent_dim)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(), nn.Linear(latent_dim // 2, input_dim),
        )
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @property
    def num_nodes(self) -> int:
        return int(self.normalized_adjacency.shape[0])

    def _validate(self, sequence: torch.Tensor) -> None:
        expected = (self.total_steps, self.num_nodes, self.input_dim)
        if tuple(sequence.shape[1:]) != expected:
            raise ValueError(
                f"Shared Teacher expected [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(sequence.shape)}"
            )

    def encode_sequence(
        self, sequence: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate(sequence)
        if mask is None:
            mask = torch.zeros(sequence.shape[:3], dtype=torch.bool, device=sequence.device)
        if tuple(mask.shape) != tuple(sequence.shape[:3]):
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match traffic tensor")

        visible = sequence.masked_fill(mask.unsqueeze(-1), 0.0)
        difference = torch.zeros_like(visible)
        difference[:, 1:] = visible[:, 1:] - visible[:, :-1]
        neighbour = torch.einsum("nm,btmc->btnc", self.normalized_adjacency, visible)
        features = torch.cat((visible, difference, neighbour, mask.unsqueeze(-1)), dim=-1)
        hidden = self.traffic_projection(features)

        graph_features = torch.cat((self.graph_pe, self.degree_feature), dim=-1)
        hidden = hidden + self.graph_projection(graph_features)[None, None]
        hidden = hidden + self.time_embedding[None, :, None]
        segments = torch.zeros(self.total_steps, dtype=torch.long, device=sequence.device)
        segments[self.input_steps:] = 1
        hidden = hidden + self.segment_embedding(segments)[None, :, None]
        hidden = hidden + mask.unsqueeze(-1) * self.mask_token
        hidden = self.input_norm(hidden)
        for block in self.blocks:
            hidden = block(hidden, self.normalized_adjacency)
        return self.output_norm(hidden)

    def reconstruct(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_head(self.encode_sequence(sequence, mask))

    def checkpoint_metadata(self) -> dict:
        return {
            "architecture": self.architecture,
            "input_steps": self.input_steps,
            "pred_steps": self.pred_steps,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "layers": self.layers,
            "heads": self.heads,
            "ff_dim": self.ff_dim,
            "graph_pe_dim": self.graph_pe_dim,
            "nodes": self.num_nodes,
        }


def build_shared_teacher(adjacency: np.ndarray, metadata: dict) -> SharedTrafficTeacher:
    return SharedTrafficTeacher(
        adjacency=adjacency,
        input_steps=int(metadata["input_steps"]),
        pred_steps=int(metadata["pred_steps"]),
        input_dim=int(metadata.get("input_dim", 1)),
        latent_dim=int(metadata["latent_dim"]),
        layers=int(metadata["layers"]),
        heads=int(metadata["heads"]),
        ff_dim=int(metadata["ff_dim"]),
        graph_pe_dim=int(metadata["graph_pe_dim"]),
        dropout=float(metadata.get("dropout", 0.1)),
    )
