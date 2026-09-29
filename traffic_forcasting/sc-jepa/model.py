"""SC-JEPA variants adapted to continuous traffic forecasting.

``scjepa_fine_only_v1`` preserves the earlier crossed-module ablation.
``scjepa_paper_full_v1`` restores the complete pre-training graph and objective
from the SC-JEPA paper; the linear forecast head remains a traffic-only adapter.
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
    encoder_bottleneck: int = 256
    predictor_dim: int = 128
    predictor_heads: int = 4
    predictor_layers: int = 2
    dropout: float = 0.1
    code_temperature: float = 0.1
    prediction_temperature: float = 0.8
    ema: float = 0.996
    fine_weight: float = 1.0
    coarse_weight: float = 0.5
    latent_weight: float = 0.1
    embedding_weight: float = 1.0
    commitment_weight: float = 0.25
    sample_entropy_weight: float = 0.005
    batch_entropy_weight: float = 0.01

    def __post_init__(self):
        if self.architecture not in ('scjepa_fine_only_v1', 'scjepa_paper_full_v1'):
            raise ValueError(f'Unsupported architecture: {self.architecture}')
        dimensions = (self.seq_len, self.pred_len, self.patch_len, self.dim,
                      self.codes, self.heads, self.layers, self.encoder_hidden,
                      self.encoder_bottleneck,
                      self.predictor_dim, self.predictor_heads, self.predictor_layers)
        if min(dimensions) <= 0:
            raise ValueError('Model dimensions must be positive')
        if self.seq_len % self.patch_len or self.pred_len % self.patch_len:
            raise ValueError('seq_len and pred_len must be divisible by patch_len')
        if self.seq_len != self.pred_len:
            raise ValueError('SC-JEPA requires equal context/target patch counts')
        if self.dim % self.heads or self.predictor_dim % self.predictor_heads:
            raise ValueError('Attention dimensions must be divisible by their head counts')
        weights = (self.fine_weight, self.coarse_weight, self.latent_weight,
                   self.embedding_weight, self.commitment_weight,
                   self.sample_entropy_weight, self.batch_entropy_weight)
        if min(weights) < 0:
            raise ValueError('Loss weights must be nonnegative')
        if not 0 <= self.dropout < 1 or not 0 <= self.ema < 1:
            raise ValueError('Invalid dropout or EMA')
        if self.code_temperature <= 0 or self.prediction_temperature <= 0:
            raise ValueError('Temperatures must be positive')


def normalize(x, paper=False):
    """Per-window RevIN statistics; the paper uses torch.std (unbiased=True)."""
    mean = x.mean(-1, keepdim=True)
    if paper:
        std = x.std(-1, keepdim=True, unbiased=True) + 1e-6
    else:
        std = (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    return (x - mean) / std, mean, std


class Encoder(nn.Module):
    """Public SC-JEPA patch MLP followed by a Transformer backbone."""
    def __init__(self, c):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(c.patch_len, c.encoder_hidden),
            nn.ReLU(inplace=False), nn.Dropout(c.dropout))
        self.feature_net = nn.Sequential(
            nn.Linear(c.encoder_hidden, c.encoder_bottleneck),
            nn.ReLU(inplace=False), nn.Dropout(c.dropout),
            nn.Linear(c.encoder_bottleneck, c.dim))
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
        features = torch.cat([self.cls_token.expand(batch, -1, -1), features], dim=1)
        features = self.transformer(features + self.pos_emb[:, :count + 1])
        return self.final_proj(features[:, 1:])


class SoftCodebook(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.entries = nn.Parameter(torch.empty(c.codes, c.dim))
        nn.init.uniform_(self.entries, -1.0 / c.codes, 1.0 / c.codes)
        self.temperature = c.code_temperature

    def forward(self, latent):
        # A larger epsilon is numerically inert at ordinary norms, but avoids
        # exploding cosine gradients if an Adam update drives a prototype
        # extremely close to the zero vector.
        codes = F.normalize(self.entries, dim=-1, eps=1e-6)
        similarity = torch.einsum(
            'bpd,kd->bpk', F.normalize(latent, dim=-1, eps=1e-6), codes)
        probabilities = F.softmax(-(1.0 - similarity) / self.temperature, dim=-1)
        aligned = torch.einsum('bpk,kd->bpd', probabilities, codes)
        return probabilities, aligned


class FinePredictor(nn.Module):
    def __init__(self, c, latent_head=False):
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
            nn.Linear(c.predictor_dim, c.predictor_dim), nn.LayerNorm(c.predictor_dim),
            nn.ReLU(inplace=False), nn.Dropout(c.dropout),
            nn.Linear(c.predictor_dim, c.codes))
        self.latent_proj = nn.Linear(c.predictor_dim, c.dim) if latent_head else None
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
        hidden = self.transformer(hidden)
        logits = self.output_proj(hidden)
        return (logits, self.latent_proj(hidden)) if self.latent_proj is not None else logits


class CoarsePredictor(nn.Module):
    """Paper coarse branch: a learned query cross-attends to all fine patches."""
    def __init__(self, c):
        super().__init__()
        patch_count = c.seq_len // c.patch_len
        self.input_proj = nn.Sequential(
            nn.Linear(c.codes, c.predictor_dim), nn.LayerNorm(c.predictor_dim),
            nn.ReLU(inplace=False), nn.Dropout(c.dropout))
        self.pos_emb = nn.Parameter(torch.randn(1, patch_count, c.predictor_dim))
        self.query_token = nn.Parameter(torch.randn(1, 1, c.predictor_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=c.predictor_dim, nhead=c.predictor_heads,
            dim_feedforward=c.predictor_dim * 2, dropout=c.dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, c.predictor_layers)
        self.cross_attention = nn.MultiheadAttention(
            c.predictor_dim, c.predictor_heads, dropout=c.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(c.predictor_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(c.predictor_dim, c.predictor_dim), nn.LayerNorm(c.predictor_dim),
            nn.ReLU(inplace=False), nn.Dropout(c.dropout), nn.Linear(c.predictor_dim, c.codes))
        self.apply(FinePredictor._initialize)

    def forward(self, probabilities):
        hidden = self.input_proj(probabilities) + self.pos_emb[:, :probabilities.shape[1]]
        hidden = self.transformer(hidden)
        query = self.query_token.expand(probabilities.shape[0], -1, -1)
        coarse, _ = self.cross_attention(query, hidden, hidden)
        return self.output_proj(self.cross_norm(coarse))


class Decoder(nn.Module):
    """The paper's auxiliary patch reconstruction decoder."""
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c.dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(c.dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(c.dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(c.dropout),
            nn.Linear(128, c.patch_len))
        self.residual_proj = nn.Linear(c.dim, c.patch_len)

    def forward(self, latent):
        return self.net(latent) + self.residual_proj(latent)


class SCJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.paper_full = config.architecture == 'scjepa_paper_full_v1'
        self.encoder = Encoder(config)
        self.quantizer = SoftCodebook(config)
        self.predictor = FinePredictor(config, latent_head=self.paper_full)
        if self.paper_full:
            self.coarse_predictor = CoarsePredictor(config)
            self.reconstructor = Decoder(config)
        self.forecast_head = nn.Linear(config.codes, config.patch_len)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_quantizer = copy.deepcopy(self.quantizer).requires_grad_(False)
        self.stage = 'pretrain'

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        self.target_quantizer.eval()
        if self.stage == 'probe':
            for module in self._online_modules():
                module.eval()
        return self

    def _online_modules(self):
        modules = [self.encoder, self.quantizer, self.predictor]
        if self.paper_full:
            modules += [self.coarse_predictor, self.reconstructor]
        return modules

    def configure(self, stage, forecast_weight=0.):
        if stage not in ('pretrain', 'probe', 'full'):
            raise ValueError(f'Unknown stage: {stage}')
        self.stage = stage
        self.requires_grad_(False)
        modules = [self.forecast_head]
        if stage == 'pretrain':
            modules += self._online_modules()
        elif stage == 'full':
            # Coarse/reconstruction branches are pre-training auxiliaries and do
            # not participate in the traffic forecast forward path.
            modules += [self.encoder, self.quantizer, self.predictor]
        for module in modules:
            module.requires_grad_(True)
        if stage == 'pretrain' and forecast_weight == 0:
            self.forecast_head.requires_grad_(False)

    def _context(self, x):
        batch, steps, nodes = x.shape
        flattened = x.transpose(1, 2).reshape(batch * nodes, steps)
        normalized, mean, std = normalize(flattened, self.paper_full)
        patches = normalized.reshape(batch * nodes, -1, self.config.patch_len)
        latent = self.encoder(patches)
        probabilities, aligned = self.quantizer(latent)
        prediction = self.predictor(probabilities)
        logits = prediction[0] if self.paper_full else prediction
        return logits, prediction, latent, probabilities, aligned, patches, mean, std, batch, nodes

    def _forecast(self, logits, mean, std, batch, nodes):
        predicted_probabilities = F.softmax(
            logits / self.config.prediction_temperature, dim=-1)
        forecast = self.forecast_head(predicted_probabilities).flatten(1) * std + mean
        return forecast.reshape(batch, nodes, self.config.pred_len).transpose(1, 2)

    def forward(self, x):
        values = self._context(x)
        return self._forecast(values[0], values[6], values[7], values[8], values[9])

    def objective(self, x, y, forecast_weight=0., reconstruction_weight=0.5):
        (logits, prediction, latent, probabilities, aligned, context_patches,
         mean, std, batch, nodes) = self._context(x)
        c = self.config
        future_raw = y.transpose(1, 2).reshape(-1, c.pred_len)
        future_norm = normalize(future_raw, self.paper_full)[0]
        future_patches = future_norm.reshape(-1, c.pred_len // c.patch_len, c.patch_len)
        with torch.no_grad():
            target_probabilities, target_aligned = self.target_quantizer(
                self.target_encoder(future_patches))
        fine_kl = F.kl_div(
            F.log_softmax(logits / c.prediction_temperature, dim=-1),
            target_probabilities, reduction='batchmean')
        losses = {'fine_kl': fine_kl}
        total = c.fine_weight * fine_kl

        if self.paper_full:
            _, latent_prediction = prediction
            patch_count = c.pred_len // c.patch_len
            coarse_raw = future_raw.reshape(-1, c.patch_len, patch_count).mean(dim=-1)
            coarse_patch = normalize(coarse_raw, paper=True)[0].unsqueeze(1)
            with torch.no_grad():
                coarse_target, _ = self.target_quantizer(self.target_encoder(coarse_patch))
            coarse_logits = self.coarse_predictor(probabilities)
            coarse_kl = F.kl_div(
                F.log_softmax(coarse_logits / c.prediction_temperature, dim=-1),
                coarse_target, reduction='batchmean')
            latent_mse = F.mse_loss(latent_prediction, target_aligned.detach())
            embedding = F.mse_loss(aligned, latent.detach())
            commitment = F.mse_loss(latent, aligned.detach())
            safe = probabilities.clamp(1e-6, 1 - 1e-6)
            sample_entropy = -(safe * safe.log()).sum(-1).mean()
            average = safe.mean(dim=(0, 1))
            negative_batch_entropy = (average * average.log()).sum()
            reconstructed = self.reconstructor(aligned).flatten(1) * std + mean
            context_raw = x.transpose(1, 2).reshape(-1, c.seq_len)
            reconstruction = F.mse_loss(reconstructed, context_raw)
            losses.update(coarse_kl=coarse_kl, latent_mse=latent_mse,
                          embedding=embedding, commitment=commitment,
                          sample_entropy=sample_entropy,
                          negative_batch_entropy=negative_batch_entropy,
                          reconstruction=reconstruction,
                          codebook_min_norm=self.quantizer.entries.norm(dim=-1).min().detach(),
                          latent_rms=latent.square().mean().sqrt().detach())
            total = (total + c.coarse_weight * coarse_kl + c.latent_weight * latent_mse
                     + c.embedding_weight * embedding + c.commitment_weight * commitment
                     + c.sample_entropy_weight * sample_entropy
                     + c.batch_entropy_weight * negative_batch_entropy
                     + reconstruction_weight * reconstruction)

        if forecast_weight > 0:
            forecast = self._forecast(logits, mean, std, batch, nodes)
            forecast_mse = F.mse_loss(forecast, y)
            losses['forecast_mse'] = forecast_mse
            total = total + forecast_weight * forecast_mse
        return total, losses

    @torch.no_grad()
    def update_ema(self):
        for online, target in ((self.encoder, self.target_encoder),
                               (self.quantizer, self.target_quantizer)):
            for parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.lerp_(parameter, 1 - self.config.ema)
