from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"


def _import_from(module: str, root: Path):
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.pop(0)


class SALTBackbone(nn.Module):
    latent_dim: int
    input_steps: int
    pred_steps: int

    def __init__(self, input_steps: int, pred_steps: int):
        super().__init__()
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.future_token = nn.Parameter(torch.zeros(1))

    def masked_sequence(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[1] != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} history steps, got {history.shape[1]}")
        future = self.future_token.to(history).expand(
            history.shape[0], self.pred_steps, history.shape[2], history.shape[3]
        )
        return torch.cat((history, future), dim=1)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(self.masked_sequence(history))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forecast_from_sequence(self.masked_sequence(history))

    def align_teacher(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        if teacher_hidden.shape[1] != self.input_steps + self.pred_steps:
            raise ValueError("Teacher and Student cover different sequence lengths")
        return teacher_hidden

    def encoder_parameters(self):
        raise NotImplementedError

    def head_parameters(self):
        raise NotImplementedError


class PatchTSTAdapter(SALTBackbone):
    """Official PatchTST encoder with its original shared flatten head."""

    def __init__(self, nodes: int, input_steps=12, pred_steps=12, d_model=128,
                 patch_len=4, stride=2, dropout=0.1):
        super().__init__(input_steps, pred_steps)
        module = _import_from("layers.PatchTST_backbone", UPSTREAM / "PatchTST")
        total = input_steps + pred_steps
        self.core = module.PatchTST_backbone(
            c_in=nodes, context_window=total, target_window=pred_steps,
            patch_len=patch_len, stride=stride, n_layers=3, d_model=d_model,
            n_heads=8, d_ff=256, norm="BatchNorm", attn_dropout=0.0,
            dropout=dropout, head_dropout=0.0, padding_patch=None,
            individual=False, revin=True, affine=True,
        )
        self.latent_dim = d_model
        self.patch_len = patch_len
        self.stride = stride

    def _normalized_patches(self, sequence):
        values = sequence[..., 0].permute(0, 2, 1)
        if self.core.revin:
            values = self.core.revin_layer(values.permute(0, 2, 1), "norm").permute(0, 2, 1)
        values = values.unfold(-1, self.patch_len, self.stride).permute(0, 1, 3, 2)
        return values

    def encode_sequence(self, sequence):
        latent = self.core.backbone(self._normalized_patches(sequence))
        return latent.permute(0, 3, 1, 2)  # B, patches, nodes, dim

    def forecast_from_sequence(self, sequence):
        latent = self.encode_sequence(sequence).permute(0, 2, 3, 1)
        prediction = self.core.head(latent)
        if self.core.revin:
            prediction = self.core.revin_layer(prediction.permute(0, 2, 1), "denorm").permute(0, 2, 1)
        return prediction.permute(0, 2, 1).unsqueeze(-1)

    def align_teacher(self, teacher_hidden):
        patches = teacher_hidden.unfold(1, self.patch_len, self.stride)
        return patches.mean(dim=-1)

    def encoder_parameters(self):
        yield self.future_token
        yield from self.core.backbone.parameters()
        if self.core.revin:
            yield from self.core.revin_layer.parameters()

    def head_parameters(self):
        yield from self.core.head.parameters()


class STGformerAdapter(SALTBackbone):
    """Official STGformer; its temporal encoder is pooled to one token per node."""

    def __init__(self, nodes: int, input_steps=12, pred_steps=12, dropout=0.1):
        super().__init__(input_steps, pred_steps)
        cls = _import_from("STGformer", UPSTREAM / "STGformer").STGformer
        total = input_steps + pred_steps
        supports = [torch.eye(nodes)]
        self.core = cls(
            num_nodes=nodes, in_steps=total, out_steps=pred_steps,
            input_dim=1, output_dim=1, input_embedding_dim=24,
            tod_embedding_dim=0, dow_embedding_dim=0,
            spatial_embedding_dim=0, adaptive_embedding_dim=12,
            num_heads=4, supports=supports, num_layers=3, dropout=dropout,
            dropout_a=0.1, kernel_size=[1],
        )
        self.latent_dim = self.core.model_dim

    def encode_sequence(self, sequence):
        core = self.core
        batch = sequence.shape[0]
        hidden = core.input_proj(sequence[..., :1])
        adaptive = core.adaptive_embedding.expand(batch, *core.adaptive_embedding.shape)
        hidden = torch.cat((hidden, core.dropout(adaptive)), dim=-1)
        hidden = core.temporal_proj(hidden.transpose(1, 3)).transpose(1, 3)
        graph = torch.matmul(core.adaptive_embedding, core.adaptive_embedding.transpose(1, 2))
        graph = core.pooling(graph.transpose(0, 2)).transpose(0, 2)
        graph = F.softmax(F.relu(graph), dim=-1)
        for attention in core.attn_layers_s:
            hidden = attention(hidden, graph)
        hidden = core.encoder_proj(hidden.transpose(1, 2).flatten(-2))
        for layer in core.encoder:
            hidden = hidden + layer(hidden)
        return hidden.unsqueeze(1)  # B, 1, nodes, dim

    def forecast_from_sequence(self, sequence):
        hidden = self.encode_sequence(sequence).squeeze(1)
        output = self.core.output_proj(hidden).view(
            hidden.shape[0], self.core.num_nodes, self.pred_steps, 1
        )
        return output.transpose(1, 2)

    def align_teacher(self, teacher_hidden):
        return teacher_hidden.mean(dim=1, keepdim=True)

    def encoder_parameters(self):
        yield self.future_token
        for name, parameter in self.core.named_parameters():
            if not name.startswith("output_proj"):
                yield parameter

    def head_parameters(self):
        yield from self.core.output_proj.parameters()


class STLAformerAdapter(SALTBackbone):
    """Official STLAformer encoder separated from its linear output layer."""

    def __init__(self, nodes: int, input_steps=12, pred_steps=12, dropout=0.1):
        super().__init__(input_steps, pred_steps)
        cls = _import_from("STLformer", UPSTREAM / "STLAformer").STLformer
        total = input_steps + pred_steps
        self.core = cls(
            num_nodes=nodes, in_steps=total, out_steps=pred_steps,
            input_dim=1, output_dim=1, input_embedding_dim=32,
            tow_embedding_dim=0, feed_forward_dim=256,
            num_heads=4, num_layers=6, dropout=dropout,
        )
        self.latent_dim = self.core.model_dim

    def encode_sequence(self, sequence):
        core = self.core
        hidden = core.input_proj(sequence[..., :1])
        node = core.node_position_encoding(hidden)
        time = core.time_position_encoding(hidden)
        baseline = core.MLP(hidden)
        hidden = hidden - baseline
        for attention in core.attn_layers_st:
            hidden = attention(hidden, node, time)
        return hidden + baseline

    def forecast_from_sequence(self, sequence):
        hidden = self.encode_sequence(sequence)
        batch = hidden.shape[0]
        output = self.core.output_proj(
            hidden.transpose(1, 2).reshape(batch, self.core.num_nodes, -1)
        )
        return output.view(batch, self.core.num_nodes, self.pred_steps, 1).transpose(1, 2)

    def encoder_parameters(self):
        yield self.future_token
        for name, parameter in self.core.named_parameters():
            if not name.startswith("output_proj"):
                yield parameter

    def head_parameters(self):
        yield from self.core.output_proj.parameters()


def build_backbone(name: str, nodes: int, input_steps=12, pred_steps=12,
                   dropout=0.1) -> SALTBackbone:
    name = name.lower()
    if name == "patchtst":
        return PatchTSTAdapter(nodes, input_steps, pred_steps, dropout=dropout)
    if name == "stgformer":
        return STGformerAdapter(nodes, input_steps, pred_steps, dropout)
    if name == "stlaformer":
        return STLAformerAdapter(nodes, input_steps, pred_steps, dropout)
    raise ValueError(f"Unknown backbone: {name}")


def compatible_encoder_transfer(target: SALTBackbone, source_state: dict[str, torch.Tensor]):
    """Copy shape-compatible encoder tensors and reinitialize city-specific ones."""
    current = target.state_dict()
    copied, skipped = [], []
    head_prefixes = ("core.head", "core.output_proj")
    for name, value in source_state.items():
        if name.startswith(head_prefixes):
            continue
        if name in current and current[name].shape == value.shape:
            current[name] = value
            copied.append(name)
        else:
            skipped.append(name)
    target.load_state_dict(current)
    if not copied:
        raise ValueError("No compatible encoder tensors were transferred")
    return {"copied": copied, "skipped": skipped}
