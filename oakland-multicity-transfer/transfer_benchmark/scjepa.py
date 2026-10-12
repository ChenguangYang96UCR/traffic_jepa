from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def load_official_modules(repo_root: Path):
    adapter = repo_root / "sc-jepa-official-traffic" / "traffic_adapter"
    if str(adapter) not in sys.path:
        sys.path.insert(0, str(adapter))
    import common  # type: ignore
    return common


class SCJEPAForecast(nn.Module):
    """Official SC-JEPA encoder/codebook/predictor with a traffic forecast head."""

    def __init__(self, repo_root: Path, pred_steps: int, patch_len: int = 3):
        super().__init__()
        self.common = load_official_modules(repo_root)
        self.pred_steps = pred_steps
        self.patch_len = patch_len
        self.num_patches = 12 // patch_len
        modules = self.common.create_official_modules(
            self.num_patches, patch_len, self.common.OFFICIAL_CONFIG
        )
        self.encoder, self.quantizer, self.predictor = modules[:3]
        codes = self.common.OFFICIAL_CONFIG["num_codes"]
        self.forecast_head = nn.Linear(self.num_patches * codes, pred_steps)

    def load_pretrained(self, checkpoint):
        self.encoder.load_state_dict(checkpoint["encoder_online"])
        self.quantizer.load_state_dict(checkpoint["quantizer_online"])
        self.predictor.load_state_dict(checkpoint["predictor"])

    def forward(self, history):
        batch, steps, nodes, _ = history.shape
        flat = history[..., 0].transpose(1, 2).reshape(batch * nodes, steps)
        mean = flat.mean(1, keepdim=True)
        std = flat.std(1, keepdim=True).clamp_min(1e-6)
        patches = ((flat - mean) / std).reshape(
            batch * nodes, self.num_patches, self.patch_len, 1
        )
        hidden = self.encoder(patches)
        probabilities, _ = self.quantizer(hidden)
        logits, _ = self.predictor(probabilities)
        codes = torch.softmax(
            logits / self.common.OFFICIAL_CONFIG["prediction_temperature"], dim=-1
        )
        prediction = self.forecast_head(codes.flatten(1)) * std + mean
        return prediction.reshape(batch, nodes, self.pred_steps).transpose(1, 2).unsqueeze(-1)

    def encoder_parameters(self):
        yield from self.encoder.parameters()
        yield from self.quantizer.parameters()
        yield from self.predictor.parameters()

    def head_parameters(self):
        yield from self.forecast_head.parameters()
