from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class AxisSelfAttention(nn.Module):
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

    def forward(self, sequence):
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        sequence = self.norm1(sequence + self.dropout(attended))
        return self.norm2(sequence + self.dropout(self.ffn(sequence)))


class STAEformerTeacherEncoder(nn.Module):
    """State-dict-compatible copy of the existing SALT Teacher encoder."""

    def __init__(self, num_nodes, max_steps, input_dim=1, input_embedding_dim=16,
                 step_embedding_dim=16, sensor_embedding_dim=0,
                 feed_forward_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.input_dim = input_dim
        self.input_embedding_dim = input_embedding_dim
        self.step_embedding_dim = step_embedding_dim
        self.sensor_embedding_dim = sensor_embedding_dim
        self.model_dim = input_embedding_dim + step_embedding_dim + sensor_embedding_dim
        self.input_projection = nn.Linear(input_dim, input_embedding_dim)
        self.mask_token = nn.Parameter(torch.zeros(input_embedding_dim))
        if step_embedding_dim:
            self.relative_step_embedding = nn.Parameter(torch.empty(max_steps, step_embedding_dim))
        if sensor_embedding_dim:
            self.sensor_embedding = nn.Parameter(torch.empty(num_nodes, sensor_embedding_dim))
        self.temporal_layers = nn.ModuleList([
            AxisSelfAttention(self.model_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            AxisSelfAttention(self.model_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, values):
        batch, steps, nodes, channels = values.shape
        if channels != self.input_dim or steps > self.max_steps:
            raise ValueError(f"Teacher cannot encode {tuple(values.shape)}")
        if self.sensor_embedding_dim and nodes != self.num_nodes:
            raise ValueError("Node-specific Teacher cannot be used across cities")
        features = [self.input_projection(values)]
        if self.step_embedding_dim:
            features.append(self.relative_step_embedding[:steps][None, :, None, :].expand(batch, -1, nodes, -1))
        if self.sensor_embedding_dim:
            features.append(self.sensor_embedding[None, None, :, :].expand(batch, steps, -1, -1))
        hidden = torch.cat(features, dim=-1)
        for layer in self.temporal_layers:
            temporal = hidden.permute(0, 2, 1, 3).reshape(batch * nodes, steps, self.model_dim)
            temporal = layer(temporal)
            hidden = temporal.reshape(batch, nodes, steps, self.model_dim).permute(0, 2, 1, 3)
        for layer in self.spatial_layers:
            hidden = layer(hidden.reshape(batch * steps, nodes, self.model_dim)).reshape(
                batch, steps, nodes, self.model_dim
            )
        return hidden


def load_frozen_teacher(path: str | Path, device: torch.device):
    saved = torch.load(path, map_location="cpu")
    if "encoder" not in saved:
        raise ValueError(f"{path} is not a SALT Teacher checkpoint")
    args = saved.get("args", {})
    state = saved["encoder"]
    step = state.get("relative_step_embedding")
    max_steps = int(step.shape[0]) if step is not None else int(args.get("input_steps", 12) + args.get("pred_steps", 12))
    sensor = state.get("sensor_embedding")
    sensor_dim = int(sensor.shape[-1]) if sensor is not None else int(args.get("teacher_sensor_dim", 0))
    nodes = int(sensor.shape[0]) if sensor is not None else int(saved.get("teacher_nodes", 1))
    teacher = STAEformerTeacherEncoder(
        num_nodes=nodes,
        max_steps=max_steps,
        input_dim=int(state["input_projection.weight"].shape[1]),
        input_embedding_dim=int(args.get("teacher_input_dim", state["input_projection.weight"].shape[0])),
        step_embedding_dim=int(args.get("teacher_step_dim", step.shape[1] if step is not None else 0)),
        sensor_embedding_dim=sensor_dim,
        feed_forward_dim=int(args.get("teacher_ff_dim", 128)),
        num_heads=int(args.get("teacher_heads", 4)),
        num_layers=int(args.get("teacher_layers", len([k for k in state if k.endswith("attention.in_proj_weight")]) // 2)),
        dropout=float(args.get("dropout", 0.1)),
    )
    teacher.load_state_dict(state)
    teacher.checkpoint_args = dict(args)
    if teacher.sensor_embedding_dim:
        raise ValueError("Cross-city runs require a node-agnostic Teacher (--teacher-sensor-dim 0)")
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher
