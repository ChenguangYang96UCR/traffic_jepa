"""SC-JEPA fine-scale framework adapted to continuous traffic forecasting.

Retained: original-style online/EMA encoders, soft codebooks and fine predictor.
Removed by experiment design: reconstruction, codebook-alignment and coarse branches.
"""
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    architecture: str = 'scjepa_fine_only_v1'
    seq_len: int = 12
    pred_len: int = 12
    patch_len: int = 3
    dim: int = 256
    codes: int = 128
    heads: int = 8
    layers: int = 6
    encoder_hidden: int = 256
    predictor_dim: int = 128
    predictor_heads: int = 4
    predictor_layers: int = 2
    dropout: float = 0.1
    code_temperature: float = 0.1
    prediction_temperature: float = 0.8
    ema: float = 0.996

    def __post_init__(self):
        if self.architecture != 'scjepa_fine_only_v1':
            raise ValueError(f'Unsupported architecture: {self.architecture}')
        dimensions = (self.seq_len, self.pred_len, self.patch_len, self.dim,
                      self.codes, self.heads, self.layers, self.encoder_hidden,
                      self.predictor_dim, self.predictor_heads, self.predictor_layers)
        if min(dimensions) <= 0:
            raise ValueError('Model dimensions must be positive')
        if self.seq_len % self.patch_len or self.pred_len % self.patch_len:
            raise ValueError('seq_len and pred_len must be divisible by patch_len')
        if self.seq_len != self.pred_len:
            raise ValueError('Original fine predictor requires equal context/target patch counts')
        if self.dim % self.heads or self.predictor_dim % self.predictor_heads:
            raise ValueError('Attention dimensions must be divisible by their head counts')
        if not 0 <= self.dropout < 1 or not 0 <= self.ema < 1:
            raise ValueError('Invalid dropout or EMA')
        if self.code_temperature <= 0 or self.prediction_temperature <= 0:
            raise ValueError('Temperatures must be positive')


def normalize(x):
    mean = x.mean(-1, keepdim=True)
    std = (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    return (x - mean) / std, mean, std


class Encoder(nn.Module):
    """Upstream encoder architecture, parameterized for short traffic patches."""
    def __init__(self, c):
        super().__init__()
        # The upstream class calls this a CNN extractor, but the current public
        # implementation uses these MLPs on each flattened patch.
        self.input_proj = nn.Sequential(
            nn.Linear(c.patch_len, c.encoder_hidden),
            nn.ReLU(inplace=False), nn.Dropout(c.dropout))
        self.feature_net = nn.Sequential(
            nn.Linear(c.encoder_hidden, c.encoder_hidden),
            nn.ReLU(inplace=False), nn.Dropout(c.dropout),
            nn.Linear(c.encoder_hidden, c.dim))
        patch_count = c.seq_len // c.patch_len
        self.cls_token = nn.Parameter(torch.randn(1, 1, c.dim))
        self.pos_emb = nn.Parameter(torch.randn(1, patch_count + 1, c.dim))
        layer = nn.TransformerEncoderLayer(
            d_model=c.dim, nhead=c.heads, dim_feedforward=c.dim * 4,
            dropout=c.dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, num_layers=c.layers)
        self.final_proj = nn.Sequential(nn.Linear(c.dim, c.dim), nn.Dropout(c.dropout))

    def forward(self, patches):
        batch, count, length = patches.shape
        features = self.feature_net(self.input_proj(patches.reshape(batch * count, length)))
        features = features.reshape(batch, count, -1)
        cls = self.cls_token.expand(batch, -1, -1)
        features = torch.cat([cls, features], dim=1)
        features = self.transformer(features + self.pos_emb[:, :count + 1])
        return self.final_proj(features[:, 1:])


class SoftCodebook(nn.Module):
    """Upstream cosine-similarity soft quantizer."""
    def __init__(self, c):
        super().__init__()
        self.entries = nn.Parameter(torch.empty(c.codes, c.dim))
        nn.init.uniform_(self.entries, -1.0 / c.codes, 1.0 / c.codes)
        self.temperature = c.code_temperature

    def forward(self, latent):
        codes = F.normalize(self.entries, dim=-1)
        similarity = torch.einsum('bpd,kd->bpk', F.normalize(latent, dim=-1), codes)
        probabilities = F.softmax(-(1.0 - similarity) / self.temperature, dim=-1)
        aligned = torch.einsum('bpk,kd->bpd', probabilities, codes)
        return probabilities, aligned


class FinePredictor(nn.Module):
    """Upstream fine Transformer predictor, with crossed-out latent head removed."""
    def __init__(self, c):
        super().__init__()
        patch_count = c.seq_len // c.patch_len
        self.input_proj = nn.Sequential(
            nn.Linear(c.codes, c.predictor_dim), nn.LayerNorm(c.predictor_dim),
            nn.ReLU(inplace=False), nn.Dropout(c.dropout))
        self.pos_emb = nn.Parameter(torch.randn(1, patch_count, c.predictor_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=c.predictor_dim, nhead=c.predictor_heads,
            dim_feedforward=c.predictor_dim * 2, dropout=c.dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, c.predictor_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(c.predictor_dim, c.predictor_dim),
            nn.LayerNorm(c.predictor_dim), nn.ReLU(inplace=False),
            nn.Dropout(c.dropout), nn.Linear(c.predictor_dim, c.codes))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, probabilities):
        hidden = self.input_proj(probabilities) + self.pos_emb[:, :probabilities.shape[1]]
        return self.output_proj(self.transformer(hidden))


class SCJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.quantizer = SoftCodebook(config)
        self.predictor = FinePredictor(config)
        # Traffic-specific downstream head consumes predicted fine code probabilities.
        self.forecast_head = nn.Linear(config.codes, config.patch_len)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_quantizer = copy.deepcopy(self.quantizer).requires_grad_(False)
        self.stage = 'pretrain'

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        self.target_quantizer.eval()
        if self.stage == 'probe':
            self.encoder.eval()
            self.quantizer.eval()
            self.predictor.eval()
        return self

    def configure(self, stage, forecast_weight=0.):
        if stage not in ('pretrain', 'probe', 'full'):
            raise ValueError(f'Unknown stage: {stage}')
        self.stage = stage
        self.requires_grad_(False)
        modules = [self.forecast_head]
        if stage != 'probe':
            modules += [self.encoder, self.quantizer, self.predictor]
        for module in modules:
            module.requires_grad_(True)
        if stage == 'pretrain' and forecast_weight == 0:
            self.forecast_head.requires_grad_(False)

    def _context_logits(self, x):
        batch, steps, nodes = x.shape
        flattened = x.transpose(1, 2).reshape(batch * nodes, steps)
        normalized, mean, std = normalize(flattened)
        patches = normalized.reshape(batch * nodes, -1, self.config.patch_len)
        latent = self.encoder(patches)
        probabilities, _ = self.quantizer(latent)
        logits = self.predictor(probabilities)
        return logits, mean, std, batch, nodes

    def _forecast(self, logits, mean, std, batch, nodes):
        predicted_probabilities = F.softmax(
            logits / self.config.prediction_temperature, dim=-1)
        forecast = self.forecast_head(predicted_probabilities).flatten(1) * std + mean
        forecast = forecast.reshape(batch, nodes, self.config.pred_len).transpose(1, 2)
        return forecast

    def forward(self, x):
        logits, mean, std, batch, nodes = self._context_logits(x)
        return self._forecast(logits, mean, std, batch, nodes)

    def objective(self, x, y, forecast_weight=0.):
        logits, mean, std, batch, nodes = self._context_logits(x)
        c = self.config
        future = y.transpose(1, 2).reshape(-1, c.pred_len)
        with torch.no_grad():
            target = normalize(future)[0].reshape(
                -1, c.pred_len // c.patch_len, c.patch_len)
            target_probabilities, _ = self.target_quantizer(
                self.target_encoder(target))
        fine_kl = F.kl_div(
            F.log_softmax(logits / c.prediction_temperature, dim=-1),
            target_probabilities, reduction='batchmean')
        losses = {'fine_kl': fine_kl}
        if forecast_weight > 0:
            forecast = self._forecast(logits, mean, std, batch, nodes)
            forecast_mse = F.mse_loss(forecast, y)
            losses['forecast_mse'] = forecast_mse
            return fine_kl + forecast_weight * forecast_mse, losses
        return fine_kl, losses

    @torch.no_grad()
    def update_ema(self):
        for online, target in ((self.encoder, self.target_encoder),
                               (self.quantizer, self.target_quantizer)):
            for parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.lerp_(parameter, 1 - self.config.ema)
