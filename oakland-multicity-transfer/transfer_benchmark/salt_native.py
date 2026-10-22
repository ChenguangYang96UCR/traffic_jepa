from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from salt_backbones.models import SALTBackbone, build_backbone

from .gnn import TrafficGNN
from .recent import DSTFGCNForecast, SDGFormerForecast
from .scjepa import SCJEPAForecast
from .tsjepa import TSJEPAForecast


SUPPORTED_TRANSFORMERS = {"pdformer", "flashst", "testam", "patchstg"}
NATIVE_BACKBONES = {"scjepa", "ts_jepa", "dcrnn", "agcrn", "sdgformer", "dstfgcn"}
SALT_TABLE_BACKBONES = (
    "pdformer", "flashst", "testam", "patchstg", "scjepa",
    "ts_jepa", "dcrnn", "agcrn", "sdgformer", "dstfgcn",
)


class NativeSALTAdapter(SALTBackbone):
    """Expose a native traffic model as a same-family SALT Teacher/Student.

    The native forecast path is unchanged.  For distillation, its spatial
    representation is projected to the SALT latent width and expanded over
    history+future positions with transferable temporal queries.  The SALT
    predictor remains external and is discarded before downstream forecasting.
    """

    def __init__(self, native: nn.Module, representation_dim: int,
                 input_steps: int, pred_steps: int, teacher_dim: int = 128):
        super().__init__(input_steps, pred_steps)
        self.native = native
        self.latent_dim = teacher_dim
        self.representation_projection = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, teacher_dim),
            nn.GELU(),
            nn.LayerNorm(teacher_dim),
        )
        self.temporal_queries = nn.Parameter(
            torch.empty(self.total_steps, teacher_dim)
        )
        self.value_projection = nn.Linear(1, teacher_dim)
        nn.init.trunc_normal_(self.temporal_queries, std=0.02)

    def _summary(self, history: torch.Tensor) -> torch.Tensor:
        hidden = self.native.encode(history)
        if hidden.ndim == 4:  # [B,T,N,D]
            hidden = hidden.mean(dim=1)
        if hidden.ndim != 3:
            raise ValueError(
                f"Native encoder must return [B,N,D] or [B,T,N,D], got {hidden.shape}"
            )
        return hidden

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(self.masked_sequence(history))

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.shape[1] != self.total_steps:
            raise ValueError(
                f"Expected {self.total_steps} sequence steps, got {sequence.shape[1]}"
            )
        # The native encoder extracts the backbone-specific context from the
        # observed history. A token projection preserves visible future values
        # for Teacher reconstruction; the Student receives future_token there.
        summary = self.representation_projection(
            self._summary(sequence[:, :self.input_steps])
        )
        values = self.value_projection(sequence[..., :1])
        return values + summary[:, None] + self.temporal_queries[None, :, None]

    def align_teacher(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        return teacher_hidden

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.native(history)

    def encoder_parameters(self):
        yield from self.native.encoder_parameters()
        yield from self.representation_projection.parameters()
        yield from self.value_projection.parameters()
        yield self.temporal_queries

    def head_parameters(self):
        yield from self.native.head_parameters()


def build_salt_table_backbone(name: str, repo_root: Path, dataset, adjacency,
                              input_steps: int, pred_steps: int,
                              dropout: float = 0.1) -> SALTBackbone:
    name = name.lower()
    if name in SUPPORTED_TRANSFORMERS:
        return build_backbone(
            name, dataset.num_nodes, input_steps, pred_steps, dropout,
            adj=adjacency, train_x=dataset.history.numpy(),
        )
    if name == "dcrnn" or name == "agcrn":
        native = TrafficGNN(
            dataset.num_nodes, input_steps, pred_steps, adjacency, name,
        )
        width = native.hidden_dim
    elif name == "scjepa":
        native = SCJEPAForecast(repo_root, pred_steps)
        width = native.representation_dim
    elif name == "ts_jepa":
        native = TSJEPAForecast(
            repo_root / "oakland-multicity-transfer" / "upstream" / "TS_JEPA",
            pred_steps, input_steps,
        )
        width = native.representation_dim
    elif name == "sdgformer":
        native = SDGFormerForecast(
            dataset.num_nodes, input_steps, pred_steps, adjacency,
            len(dataset), dropout,
        )
        width = native.representation_dim
    elif name == "dstfgcn":
        native = DSTFGCNForecast(
            dataset.num_nodes, input_steps, pred_steps, adjacency, dropout=0.2,
        )
        width = native.representation_dim
    else:
        raise ValueError(
            f"Unknown SALT table backbone {name!r}; choose from {SALT_TABLE_BACKBONES}"
        )
    return NativeSALTAdapter(native, width, input_steps, pred_steps)

