from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


REPO = Path(__file__).resolve().parents[2]
SALT_ROOT = REPO / "staeformer-salt"
if str(SALT_ROOT) not in sys.path:
    sys.path.insert(0, str(SALT_ROOT))

from staeformer_salt.model import STAEformerEncoder  # noqa: E402


class STAEformerSALTBackbone(nn.Module):
    """The d=128 STAEformer Student used by the established SALT sweep."""

    def __init__(self, nodes: int, input_steps: int, pred_steps: int,
                 dropout: float = 0.1):
        super().__init__()
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.total_steps = input_steps + pred_steps
        self.encoder = STAEformerEncoder(
            num_nodes=nodes, max_steps=self.total_steps, input_dim=1,
            input_embedding_dim=24, step_embedding_dim=24,
            sensor_embedding_dim=80, feed_forward_dim=256,
            num_heads=4, num_layers=3, dropout=dropout,
        )
        self.latent_dim = self.encoder.model_dim
        if self.latent_dim != 128:
            raise ValueError(f"Expected the established d=128 setup, got {self.latent_dim}")
        self.future_token = nn.Parameter(torch.zeros(1))
        self.output_projection = nn.Linear(self.latent_dim, 1)

    def masked_sequence(self, history):
        future = self.future_token.to(history).expand(
            history.shape[0], self.pred_steps, history.shape[2], history.shape[3]
        )
        return torch.cat((history, future), dim=1)

    def encode_sequence(self, sequence):
        return self.encoder(sequence)

    def encode_history(self, history):
        sequence = self.masked_sequence(history)
        mask = torch.zeros(
            history.shape[0], self.total_steps, dtype=torch.bool, device=history.device
        )
        mask[:, self.input_steps:] = True
        return self.encoder(sequence, step_mask=mask)

    def align_teacher(self, teacher_hidden):
        return teacher_hidden

    def forward(self, history):
        hidden = self.encode_history(history)
        return self.output_projection(hidden[:, self.input_steps:])

    def encoder_parameters(self):
        yield self.future_token
        yield from self.encoder.parameters()

    def head_parameters(self):
        yield from self.output_projection.parameters()
