from __future__ import annotations

import copy

import torch
from torch import nn


class AxisSelfAttention(nn.Module):
    """Vanilla Transformer block over either the time or sensor axis."""

    def __init__(self, dim: int, heads: int, feed_forward_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        sequence = self.norm1(sequence + self.dropout(attended))
        return self.norm2(sequence + self.dropout(self.ffn(sequence)))


class STAEformerEncoder(nn.Module):
    """STAEformer encoder for traffic windows without calendar timestamps.

    The original joint spatio-temporal adaptive table is factorized into a
    city-independent relative-step embedding and a city-specific sensor
    embedding.  Consequently, all Transformer weights and the step embedding
    can be transferred between cities with different numbers of sensors while
    the destination sensor embedding is learned from scratch.
    """

    def __init__(
        self,
        num_nodes: int,
        max_steps: int,
        input_dim: int = 1,
        input_embedding_dim: int = 24,
        step_embedding_dim: int = 24,
        sensor_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
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
        if step_embedding_dim:
            self.relative_step_embedding = nn.Parameter(
                torch.empty(max_steps, step_embedding_dim)
            )
            nn.init.xavier_uniform_(self.relative_step_embedding)
        if sensor_embedding_dim:
            self.sensor_embedding = nn.Parameter(
                torch.empty(num_nodes, sensor_embedding_dim)
            )
            nn.init.xavier_uniform_(self.sensor_embedding)
        self.temporal_layers = nn.ModuleList(
            [AxisSelfAttention(self.model_dim, num_heads, feed_forward_dim, dropout) for _ in range(num_layers)]
        )
        self.spatial_layers = nn.ModuleList(
            [AxisSelfAttention(self.model_dim, num_heads, feed_forward_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, values: torch.Tensor, start_position: int = 0) -> torch.Tensor:
        batch, steps, nodes, channels = values.shape
        if nodes != self.num_nodes or start_position + steps > self.max_steps:
            raise ValueError(
                f"Encoder expected <= {self.max_steps} steps and {self.num_nodes} nodes, "
                f"got {values.shape} at offset {start_position}"
            )
        if channels != self.input_dim:
            raise ValueError(f"Encoder expected {self.input_dim} channels, got {channels}")
        features = [self.input_projection(values)]
        if self.step_embedding_dim:
            step = self.relative_step_embedding[start_position : start_position + steps]
            features.append(step[None, :, None, :].expand(batch, -1, nodes, -1))
        if self.sensor_embedding_dim:
            features.append(
                self.sensor_embedding[None, None, :, :].expand(batch, steps, -1, -1)
            )
        hidden = torch.cat(features, dim=-1)

        for layer in self.temporal_layers:
            temporal = hidden.permute(0, 2, 1, 3).reshape(batch * nodes, steps, self.model_dim)
            temporal = layer(temporal)
            hidden = temporal.reshape(batch, nodes, steps, self.model_dim).permute(0, 2, 1, 3)
        for layer in self.spatial_layers:
            spatial = hidden.reshape(batch * steps, nodes, self.model_dim)
            spatial = layer(spatial)
            hidden = spatial.reshape(batch, steps, nodes, self.model_dim)
        return hidden


class FutureLatentPredictor(nn.Module):
    """Future queries cross-attend each sensor's history, then mix sensors."""

    def __init__(self, pred_steps: int, nodes: int, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.pred_steps = pred_steps
        self.nodes = nodes
        self.dim = dim
        self.queries = nn.Parameter(torch.empty(pred_steps, nodes, dim))
        nn.init.xavier_uniform_(self.queries)
        self.temporal_cross_attention = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) for _ in range(layers)]
        )
        self.cross_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.spatial_layers = nn.ModuleList(
            [AxisSelfAttention(dim, heads, 4 * dim, dropout) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch, history_steps, nodes, dim = context.shape
        query = self.queries[None].expand(batch, -1, -1, -1)
        memory = context.permute(0, 2, 1, 3).reshape(batch * nodes, history_steps, dim)
        for cross, norm, spatial_layer in zip(
            self.temporal_cross_attention, self.cross_norms, self.spatial_layers
        ):
            temporal_query = query.permute(0, 2, 1, 3).reshape(
                batch * nodes, self.pred_steps, dim
            )
            update, _ = cross(temporal_query, memory, memory, need_weights=False)
            temporal_query = norm(temporal_query + update)
            query = temporal_query.reshape(batch, nodes, self.pred_steps, dim).permute(0, 2, 1, 3)
            spatial = query.reshape(batch * self.pred_steps, nodes, dim)
            query = spatial_layer(spatial).reshape(batch, self.pred_steps, nodes, dim)
        return self.output_norm(query)


class FutureJEPA(nn.Module):
    def __init__(self, encoder: STAEformerEncoder, input_steps: int, pred_steps: int, heads: int, predictor_layers: int, dropout: float):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.predictor = FutureLatentPredictor(
            pred_steps,
            encoder.num_nodes,
            encoder.model_dim,
            heads,
            predictor_layers,
            dropout,
        )
        self.input_steps = input_steps
        self.pred_steps = pred_steps

    def forward(self, history: torch.Tensor, future: torch.Tensor):
        context = self.online_encoder(history, start_position=0)
        predicted_future = self.predictor(context)
        with torch.no_grad():
            # Encode the true future block only. Reusing relative positions 0:H
            # keeps the learned step table compatible with the standard
            # downstream encoder and avoids positions beyond the input window.
            target_future = self.target_encoder(future, start_position=0)
        return predicted_future, target_future.detach()

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)


class STAEformerForecast(nn.Module):
    """Original STAEformer mixed projection for direct multi-step forecasting."""

    def __init__(self, encoder: STAEformerEncoder, input_steps: int, pred_steps: int, output_dim: int = 1):
        super().__init__()
        self.encoder = encoder
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.output_dim = output_dim
        self.output_projection = nn.Linear(
            input_steps * encoder.model_dim, pred_steps * output_dim
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(history, start_position=0)
        batch, steps, nodes, dim = hidden.shape
        mixed = hidden.transpose(1, 2).reshape(batch, nodes, steps * dim)
        output = self.output_projection(mixed).view(
            batch, nodes, self.pred_steps, self.output_dim
        )
        return output.transpose(1, 2)
