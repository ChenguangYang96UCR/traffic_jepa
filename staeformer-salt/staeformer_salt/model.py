from __future__ import annotations

import torch
from torch import nn


class AxisSelfAttention(nn.Module):
    """Transformer block over either the time or sensor axis."""

    def __init__(self, dim: int, heads: int, feed_forward_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, feed_forward_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(feed_forward_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        sequence = self.norm1(sequence + self.dropout(attended))
        return self.norm2(sequence + self.dropout(self.ffn(sequence)))


class STAEformerEncoder(nn.Module):
    """Traffic encoder with transferable step and city-specific sensor embeddings."""

    def __init__(self, num_nodes: int, max_steps: int, input_dim: int = 1,
                 input_embedding_dim: int = 24, step_embedding_dim: int = 24,
                 sensor_embedding_dim: int = 80, feed_forward_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.input_dim = input_dim
        self.input_embedding_dim = input_embedding_dim
        self.step_embedding_dim = step_embedding_dim
        self.sensor_embedding_dim = sensor_embedding_dim
        self.model_dim = input_embedding_dim + step_embedding_dim + sensor_embedding_dim
        if self.model_dim % num_heads:
            raise ValueError("Total model dimension must be divisible by num_heads")
        self.input_projection = nn.Linear(input_dim, input_embedding_dim)
        self.mask_token = nn.Parameter(torch.zeros(input_embedding_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        if step_embedding_dim:
            self.relative_step_embedding = nn.Parameter(torch.empty(max_steps, step_embedding_dim))
            nn.init.xavier_uniform_(self.relative_step_embedding)
        if sensor_embedding_dim:
            self.sensor_embedding = nn.Parameter(torch.empty(num_nodes, sensor_embedding_dim))
            nn.init.xavier_uniform_(self.sensor_embedding)
        self.temporal_layers = nn.ModuleList([
            AxisSelfAttention(self.model_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            AxisSelfAttention(self.model_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, values: torch.Tensor, start_position: int = 0,
                step_mask: torch.Tensor | None = None,
                node_indices: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps, nodes, channels = values.shape
        expected_nodes = (
            nodes
            if self.sensor_embedding_dim == 0 and node_indices is None
            else self.num_nodes if node_indices is None else int(node_indices.numel())
        )
        if nodes != expected_nodes or start_position + steps > self.max_steps:
            raise ValueError(
                f"Expected <= {self.max_steps} steps and {expected_nodes} selected nodes, "
                f"got {values.shape}"
            )
        if channels != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} channels, got {channels}")
        projected = self.input_projection(values)
        if step_mask is not None:
            if step_mask.shape == (batch, steps):
                mask = step_mask[:, :, None, None]
            elif step_mask.shape == (batch, steps, nodes):
                mask = step_mask[:, :, :, None]
            else:
                raise ValueError(f"step_mask must be {(batch, steps)} or {(batch, steps, nodes)}")
            projected = torch.where(
                mask.to(values.device, torch.bool),
                self.mask_token.view(1, 1, 1, -1), projected,
            )
        features = [projected]
        if self.step_embedding_dim:
            step = self.relative_step_embedding[start_position:start_position + steps]
            features.append(step[None, :, None, :].expand(batch, -1, nodes, -1))
        if self.sensor_embedding_dim:
            sensor_embedding = self.sensor_embedding
            if node_indices is not None:
                node_indices = node_indices.to(device=values.device, dtype=torch.long)
                if node_indices.min() < 0 or node_indices.max() >= self.num_nodes:
                    raise ValueError("node_indices are outside the teacher sensor table")
                sensor_embedding = sensor_embedding.index_select(0, node_indices)
            features.append(sensor_embedding[None, None, :, :].expand(batch, steps, -1, -1))
        elif node_indices is not None:
            raise ValueError("node_indices are unnecessary for a node-agnostic encoder")
        hidden = torch.cat(features, dim=-1)
        for layer in self.temporal_layers:
            temporal = hidden.permute(0, 2, 1, 3).reshape(batch * nodes, steps, self.model_dim)
            temporal = layer(temporal)
            hidden = temporal.reshape(batch, nodes, steps, self.model_dim).permute(0, 2, 1, 3)
        for layer in self.spatial_layers:
            spatial = hidden.reshape(batch * steps, nodes, self.model_dim)
            hidden = layer(spatial).reshape(batch, steps, nodes, self.model_dim)
        return hidden


class MaskedFutureForecast(nn.Module):
    """Forecast future flow using appended learned mask positions."""

    def __init__(self, encoder: STAEformerEncoder, input_steps: int,
                 pred_steps: int, output_dim: int = 1):
        super().__init__()
        if encoder.max_steps < input_steps + pred_steps:
            raise ValueError("Encoder max_steps must cover history plus future masks")
        self.encoder = encoder
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.output_projection = nn.Linear(encoder.model_dim, output_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, channels = history.shape
        if steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} history steps, got {steps}")
        sequence = torch.cat((history, history.new_zeros(batch, self.pred_steps, nodes, channels)), dim=1)
        mask = torch.zeros(batch, self.input_steps + self.pred_steps,
                           dtype=torch.bool, device=history.device)
        mask[:, self.input_steps:] = True
        hidden = self.encoder(sequence, step_mask=mask)
        return self.output_projection(hidden[:, self.input_steps:])
