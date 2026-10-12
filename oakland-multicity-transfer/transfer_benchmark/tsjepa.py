from __future__ import annotations

import copy
import importlib
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


OFFICIAL_COMMIT = "830ee5c3a7d4e85623bd5a07992732be7fda9bc2"


def load_official_classes(official_root: Path):
    """Load TS-JEPA under a private namespace to avoid generic ``src`` clashes."""
    source = official_root / "src"
    models = source / "models"
    if not (models / "encoder.py").exists():
        raise FileNotFoundError(
            f"Official TS-JEPA was not found at {official_root}. "
            "Run: bash scripts/fetch_ts_jepa.sh"
        )
    packages = {
        "ts_jepa_official": source,
        "ts_jepa_official.models": models,
        "ts_jepa_official.models.utils": models / "utils",
    }
    for name, path in packages.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__package__ = name
            sys.modules[name] = module
    encoder = importlib.import_module("ts_jepa_official.models.encoder").Encoder
    predictor = importlib.import_module("ts_jepa_official.models.predictor").Predictor
    return encoder, predictor


def _init_official_style(module: nn.Module):
    """Match the Linear/LayerNorm initialization used by official pretrain.py."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.trunc_normal_(child.weight, std=0.02)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)


def batched_gather(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or indices.ndim != 2 or tokens.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Expected tokens [B,P,D] and indices [B,K], got "
            f"{tuple(tokens.shape)} and {tuple(indices.shape)}"
        )
    return torch.gather(
        tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    )


def random_patch_split(batch: int, patches: int, mask_ratio: float,
                       device: torch.device, generator=None):
    masked_count = min(patches - 1, max(1, int(patches * mask_ratio)))
    order = torch.rand(batch, patches, device=device, generator=generator).argsort(dim=1)
    return order[:, :masked_count], order[:, masked_count:]


def encode_visible(encoder: nn.Module, patches: torch.Tensor,
                   visible_indices: torch.Tensor) -> torch.Tensor:
    hidden = encoder.tokenizer(patches)
    positions = encoder.pos_embed.expand(hidden.shape[0], -1, -1)
    hidden = batched_gather(hidden + positions, visible_indices)
    for block in encoder.predictor_blocks:
        hidden = block(hidden, mask=None)
    return encoder.encoder_norm(hidden)


def predict_masked(predictor: nn.Module, visible: torch.Tensor,
                   masked_indices: torch.Tensor,
                   visible_indices: torch.Tensor) -> torch.Tensor:
    batch = visible.shape[0]
    hidden = predictor.predictor_embed(visible)
    positions = predictor.pos_embed.expand(batch, -1, -1)
    hidden = hidden + batched_gather(positions, visible_indices)
    targets = predictor.mask_token.expand(batch, masked_indices.shape[1], -1)
    targets = targets + batched_gather(positions, masked_indices)
    hidden = torch.cat((hidden, targets), dim=1)
    for block in predictor.predictor_blocks:
        hidden = block(hidden, mask=None)
    hidden = predictor.predictor_norm(hidden[:, -masked_indices.shape[1]:])
    return predictor.predictor_proj(hidden)


class TSJEPAForecast(nn.Module):
    """Official TS-JEPA encoder with a fresh horizon-specific traffic head."""

    def __init__(self, official_root: Path, pred_steps: int, input_steps: int = 12,
                 patch_size: int = 3, embed_dim: int = 128):
        super().__init__()
        if input_steps % patch_size:
            raise ValueError("TS-JEPA requires input_steps divisible by patch_size")
        Encoder, _ = load_official_classes(official_root)
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.patch_size = patch_size
        self.num_patches = input_steps // patch_size
        self.embed_dim = embed_dim
        self.encoder = Encoder(
            num_patches=self.num_patches, dim_in=patch_size,
            kernel_size=patch_size, embed_dim=embed_dim, embed_bias=True,
            nhead=2, num_layers=1, jepa=True,
        )
        self.forecast_head = nn.Linear(self.num_patches * embed_dim, pred_steps)
        _init_official_style(self.encoder)
        _init_official_style(self.forecast_head)

    def patches(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4 or history.shape[1] != self.input_steps:
            raise ValueError(
                f"Expected history [B,{self.input_steps},N,1], got {tuple(history.shape)}"
            )
        batch, _, nodes, _ = history.shape
        return history[..., 0].permute(0, 2, 1).reshape(
            batch * nodes, self.num_patches, self.patch_size
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch, _, nodes, _ = history.shape
        hidden = self.encoder(self.patches(history))
        prediction = self.forecast_head(hidden.flatten(1))
        return prediction.reshape(batch, nodes, self.pred_steps).transpose(1, 2).unsqueeze(-1)

    def load_pretrained(self, state):
        self.encoder.load_state_dict(state)

    def encoder_parameters(self):
        yield from self.encoder.parameters()

    def head_parameters(self):
        yield from self.forecast_head.parameters()


class TSJEPAPretrainer(nn.Module):
    def __init__(self, official_root: Path, input_steps=12, patch_size=3,
                 embed_dim=128):
        super().__init__()
        if input_steps % patch_size:
            raise ValueError("TS-JEPA requires input_steps divisible by patch_size")
        Encoder, Predictor = load_official_classes(official_root)
        self.input_steps = input_steps
        self.patch_size = patch_size
        self.num_patches = input_steps // patch_size
        self.online = Encoder(
            num_patches=self.num_patches, dim_in=patch_size,
            kernel_size=patch_size, embed_dim=embed_dim, embed_bias=True,
            nhead=2, num_layers=1, jepa=True,
        )
        self.predictor = Predictor(
            num_patches=self.num_patches, encoder_embed_dim=embed_dim,
            predictor_embed_dim=128, nhead=2, num_layers=1,
        )
        _init_official_style(self.online)
        _init_official_style(self.predictor)
        self.target = copy.deepcopy(self.online)
        for parameter in self.target.parameters():
            parameter.requires_grad = False

    def patches(self, history: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = history.shape
        if steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} history steps, got {steps}")
        return history[..., 0].permute(0, 2, 1).reshape(
            batch * nodes, self.num_patches, self.patch_size
        )

    def loss(self, history: torch.Tensor, mask_ratio: float, generator=None):
        patches = self.patches(history)
        masked, visible = random_patch_split(
            patches.shape[0], self.num_patches, mask_ratio,
            patches.device, generator,
        )
        with torch.no_grad():
            target = F.layer_norm(
                self.target(patches), (self.target.embed_dim,)
            )
            target = batched_gather(target, masked)
        context = encode_visible(self.online, patches, visible)
        prediction = predict_masked(self.predictor, context, masked, visible)
        return F.l1_loss(prediction, target)

    @torch.no_grad()
    def update_target(self, momentum: float):
        for online, target in zip(self.online.parameters(), self.target.parameters()):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)


@torch.no_grad()
def evaluate_ts_jepa(model: TSJEPAPretrainer, loader, device,
                     mask_ratio: float, seed: int) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = count = 0
    for history, _ in loader:
        history = history.to(device)
        loss = model.loss(history, mask_ratio, generator)
        items = history.shape[0] * history.shape[2]
        total += float(loss) * items
        count += items
    return total / count


def fit_ts_jepa(official_root: Path, loaders, device, output: Path,
                 epochs: int, lr: float, patience: int, mask_ratio: float,
                 ema_momentum: float, seed: int, input_steps=12,
                 patch_size=3):
    output.mkdir(parents=True, exist_ok=True)
    model = TSJEPAPretrainer(
        official_root, input_steps=input_steps, patch_size=patch_size,
    ).to(device)
    parameters = list(model.online.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.5,
        total_iters=max(1, epochs),
    )
    checkpoint = output / "best_ts_jepa.pt"
    best, stale, log = math.inf, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = count = 0
        momentum = ema_momentum + (epoch - 1) * (1.0 - ema_momentum) / max(1, epochs * 1.25)
        for history, _ in loaders["train"]:
            history = history.to(device)
            loss = model.loss(history, mask_ratio)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite TS-JEPA loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            model.update_target(momentum)
            items = history.shape[0] * history.shape[2]
            total += float(loss.detach()) * items
            count += items
        validation = evaluate_ts_jepa(
            model, loaders["val"], device, mask_ratio, seed + 97_000,
        )
        row = {
            "epoch": epoch, "train_latent_l1": total / count,
            "val_latent_l1": validation, "ema_momentum": momentum,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log.append(row)
        print(
            f"ts_jepa epoch {epoch:03d}: train={total / count:.7f} "
            f"val={validation:.7f} ema={momentum:.6f}", flush=True,
        )
        if validation < best:
            best, stale = validation, 0
            torch.save({
                "encoder": model.online.state_dict(), "epoch": epoch,
                "val_latent": validation, "official_commit": OFFICIAL_COMMIT,
                "input_steps": input_steps, "patch_size": patch_size,
                "mask_ratio": mask_ratio, "ema_momentum": ema_momentum,
            }, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
        scheduler.step()
    saved = torch.load(checkpoint, map_location="cpu")
    (output / "history.json").write_text(json.dumps(log, indent=2) + "\n")
    (output / "pretrain.json").write_text(json.dumps({
        key: value for key, value in saved.items() if key != "encoder"
    }, indent=2) + "\n")
    return saved
