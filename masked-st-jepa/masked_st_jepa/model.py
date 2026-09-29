from __future__ import annotations

import copy

import torch
from torch import nn


class SpatialLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.self_projection = nn.Linear(dim, dim)
        self.neighbor_projection = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("nm,btmd->btnd", adjacency, x)
        update = torch.nn.functional.gelu(
            self.self_projection(x) + self.neighbor_projection(neighbors)
        )
        return self.norm(x + self.dropout(update))


class STBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.spatial = SpatialLayer(dim, dropout)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.temporal_norm = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor, causal: bool
    ) -> torch.Tensor:
        x = self.spatial(x, adjacency)
        batch, steps, nodes, dim = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, dim)
        causal_mask = None
        if causal:
            causal_mask = torch.triu(
                torch.ones(steps, steps, device=x.device, dtype=torch.bool), diagonal=1
            )
        sequence = self.temporal(sequence, mask=causal_mask)
        sequence = self.temporal_norm(sequence)
        return sequence.reshape(batch, nodes, steps, dim).permute(0, 2, 1, 3)


class STEncoder(nn.Module):
    def __init__(
        self,
        nodes: int,
        steps: int,
        dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.value_embedding = nn.Linear(1, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.mask_embedding = nn.Embedding(2, dim)
        self.sensor_embedding = nn.Embedding(nodes, dim)
        self.time_embedding = nn.Embedding(steps, dim)
        self.blocks = nn.ModuleList(
            [STBlock(dim, heads, dropout) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        batch, steps, nodes, _ = values.shape
        embedded = self.value_embedding(values)
        if mask is None:
            mask = torch.zeros(batch, steps, nodes, dtype=torch.bool, device=values.device)
        embedded = torch.where(mask[..., None], self.mask_token, embedded)
        sensor_ids = torch.arange(nodes, device=values.device)
        time_ids = torch.arange(steps, device=values.device)
        embedded = (
            embedded
            + self.sensor_embedding(sensor_ids)[None, None, :, :]
            + self.time_embedding(time_ids)[None, :, None, :]
            + self.mask_embedding(mask.long())
        )
        for block in self.blocks:
            embedded = block(embedded, adjacency, causal)
        return self.output_norm(embedded)


class MaskedSTJEPA(nn.Module):
    def __init__(
        self,
        nodes: int,
        steps: int,
        adjacency: torch.Tensor,
        dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.online_encoder = STEncoder(nodes, steps, dim, heads, layers, dropout)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim)
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )
        self.register_buffer("adjacency", adjacency)
        self.causal = causal

    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        context = self.online_encoder(values, self.adjacency, mask, self.causal)
        predicted_latent = self.predictor(context)
        predicted_value = self.value_head(predicted_latent)
        with torch.no_grad():
            target_latent = self.target_encoder(values, self.adjacency, None, self.causal)
        return predicted_value, predicted_latent, target_latent.detach()

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for target, online in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters()
        ):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

    def set_finetune_strategy(self, strategy: str) -> None:
        if strategy == "full":
            for parameter in self.online_encoder.parameters():
                parameter.requires_grad = True
            for parameter in self.predictor.parameters():
                parameter.requires_grad = True
        elif strategy == "head":
            for parameter in self.online_encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.predictor.parameters():
                parameter.requires_grad = False
        else:
            raise ValueError(f"Unknown fine-tune strategy: {strategy}")
        for parameter in self.value_head.parameters():
            parameter.requires_grad = True

