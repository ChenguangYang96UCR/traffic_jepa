from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .graph import laplacian_pe, pattern_keys, profile_distance, shortest_hops, spectral_patch_indices

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"


def _file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_from(module: str, root: Path):
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.pop(0)


class SALTBackbone(nn.Module):
    """Contract shared by scratch, supervised, SALT and JEPA runs."""

    latent_dim: int

    def __init__(self, input_steps: int, pred_steps: int):
        super().__init__()
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.total_steps = input_steps + pred_steps
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
        if teacher_hidden.shape[1] != self.total_steps:
            raise ValueError("Teacher and Student cover different sequence lengths")
        return teacher_hidden

    def encoder_parameters(self):
        raise NotImplementedError

    def head_parameters(self):
        raise NotImplementedError


def _parameters_except(module: nn.Module, excluded_prefixes: tuple[str, ...]):
    for name, parameter in module.named_parameters():
        if not name.startswith(excluded_prefixes):
            yield parameter


class PatchTSTAdapter(SALTBackbone):
    def __init__(self, nodes, input_steps=12, pred_steps=12, dropout=0.1, **_):
        super().__init__(input_steps, pred_steps)
        module = _import_from("layers.PatchTST_backbone", UPSTREAM / "PatchTST")
        self.core = module.PatchTST_backbone(
            c_in=nodes, context_window=self.total_steps, target_window=pred_steps,
            patch_len=4, stride=2, n_layers=3, d_model=128, n_heads=8,
            d_ff=256, norm="BatchNorm", attn_dropout=0.0, dropout=dropout,
            head_dropout=0.0, padding_patch=None, individual=False,
            revin=True, affine=True,
        )
        self.latent_dim = 128
        self.patch_len, self.stride = 4, 2

    def _patches(self, sequence):
        values = sequence[..., 0].permute(0, 2, 1)
        if self.core.revin:
            values = self.core.revin_layer(values.permute(0, 2, 1), "norm").permute(0, 2, 1)
        return values.unfold(-1, self.patch_len, self.stride).permute(0, 1, 3, 2)

    def encode_sequence(self, sequence):
        return self.core.backbone(self._patches(sequence)).permute(0, 3, 1, 2)

    def forecast_from_sequence(self, sequence):
        hidden = self.encode_sequence(sequence).permute(0, 2, 3, 1)
        prediction = self.core.head(hidden)
        if self.core.revin:
            prediction = self.core.revin_layer(
                prediction.permute(0, 2, 1), "denorm"
            ).permute(0, 2, 1)
        return prediction.permute(0, 2, 1).unsqueeze(-1)

    def align_teacher(self, teacher_hidden):
        return teacher_hidden.unfold(1, self.patch_len, self.stride).mean(dim=-1)

    def encoder_parameters(self):
        yield self.future_token
        yield from self.core.backbone.parameters()
        if self.core.revin:
            yield from self.core.revin_layer.parameters()

    def head_parameters(self):
        yield from self.core.head.parameters()


class STAEformerAdapter(SALTBackbone):
    def __init__(self, nodes, input_steps=12, pred_steps=12, dropout=0.1, **_):
        super().__init__(input_steps, pred_steps)
        cls = _file_module(
            "official_staeformer", UPSTREAM / "STAEformer" / "STAEformer.py"
        ).STAEformer
        self.core = cls(
            num_nodes=nodes, in_steps=self.total_steps, out_steps=pred_steps,
            steps_per_day=288, input_dim=1, output_dim=1,
            input_embedding_dim=24, tod_embedding_dim=0, dow_embedding_dim=0,
            spatial_embedding_dim=0, adaptive_embedding_dim=80,
            feed_forward_dim=256, num_heads=4, num_layers=3,
            dropout=dropout, use_mixed_proj=True,
        )
        self.latent_dim = self.core.model_dim

    def encode_sequence(self, sequence):
        batch = sequence.shape[0]
        hidden = self.core.input_proj(sequence[..., :1])
        adaptive = self.core.adaptive_embedding.expand(batch, *self.core.adaptive_embedding.shape)
        hidden = torch.cat((hidden, adaptive), dim=-1)
        for layer in self.core.attn_layers_t:
            hidden = layer(hidden, dim=1)
        for layer in self.core.attn_layers_s:
            hidden = layer(hidden, dim=2)
        return hidden

    def forecast_from_sequence(self, sequence):
        hidden = self.encode_sequence(sequence)
        batch = hidden.shape[0]
        output = self.core.output_proj(
            hidden.transpose(1, 2).reshape(batch, self.core.num_nodes, -1)
        )
        return output.view(batch, self.core.num_nodes, self.pred_steps, 1).transpose(1, 2)

    def encoder_parameters(self):
        yield self.future_token
        yield from _parameters_except(self.core, ("output_proj",))

    def head_parameters(self):
        yield from self.core.output_proj.parameters()


class PDFormerAdapter(SALTBackbone):
    def __init__(self, nodes, input_steps=12, pred_steps=12, dropout=0.1,
                 adj=None, train_x=None, feature_dim=1, **_):
        super().__init__(input_steps, pred_steps)
        if adj is None or train_x is None:
            raise ValueError("PDFormer requires adjacency and training histories")
        pd_root = UPSTREAM / "PDFormer"
        sys.path.insert(0, str(pd_root))
        try:
            cls = importlib.import_module(
                "libcity.model.traffic_flow_prediction.PDFormer"
            ).PDFormer
        finally:
            sys.path.pop(0)
        config = {
            "dataset": "released_city_windows", "device": torch.device("cpu"),
            "input_window": self.total_steps, "output_window": pred_steps,
            "output_dim": 1, "embed_dim": 64, "skip_dim": 256,
            "lape_dim": 8, "geo_num_heads": 4, "sem_num_heads": 2,
            "t_num_heads": 2, "mlp_ratio": 4, "qkv_bias": True,
            "drop": dropout, "attn_drop": 0.0, "drop_path": 0.1,
            "s_attn_size": 3, "t_attn_size": 3, "enc_depth": 4,
            "type_ln": "pre", "type_short_path": "hop",
            "add_time_in_day": False, "add_day_in_week": False,
            "far_mask_delta": 5, "dtw_delta": 5,
            "use_curriculum_learning": False, "step_size": 1,
        }
        features = {
            "scaler": None, "num_nodes": nodes, "feature_dim": feature_dim,
            "ext_dim": 0, "num_batches": max(1, math.ceil(len(train_x) / 16)),
            "dtw_matrix": profile_distance(train_x), "adj_mx": adj,
            "sd_mx": None, "sh_mx": torch.from_numpy(shortest_hops(adj)),
            "pattern_keys": pattern_keys(train_x),
        }
        self.register_buffer("lap", torch.from_numpy(laplacian_pe(adj, 8)))
        self.core = cls(config, features)
        for name in ("geo_mask", "sem_mask", "pattern_keys"):
            value = getattr(self.core, name)
            delattr(self.core, name)
            self.core.register_buffer(name, value)
        self.latent_dim = self.core.embed_dim

    def _patterns(self, sequence):
        core, steps = self.core, sequence.shape[1]
        windows = [
            F.pad(
                sequence[:, :steps + i + 1 - core.s_attn_size, :, :core.output_dim],
                (0, 0, 0, 0, core.s_attn_size - 1 - i, 0),
            ).unsqueeze(-2)
            for i in range(core.s_attn_size)
        ]
        raw = torch.cat(windows, dim=-2)
        queries, keys = [], []
        for i in range(core.output_dim):
            queries.append(core.pattern_embeddings[i](raw[..., i]).unsqueeze(-1))
            keys.append(core.pattern_embeddings[i](core.pattern_keys[..., i]).unsqueeze(-1))
        return torch.cat(queries, -1), torch.cat(keys, -1)

    def encode_sequence(self, sequence):
        patterns, keys = self._patterns(sequence)
        hidden = self.core.enc_embed_layer(sequence, self.lap)
        for block in self.core.encoder_blocks:
            hidden = block(hidden, patterns, keys, self.core.geo_mask, self.core.sem_mask)
        return hidden

    def forecast_from_sequence(self, sequence):
        return self.core({"X": sequence}, self.lap)

    def encoder_parameters(self):
        yield self.future_token
        yield from self.core.pattern_embeddings.parameters()
        yield from self.core.enc_embed_layer.parameters()
        yield from self.core.encoder_blocks.parameters()

    def head_parameters(self):
        yield from self.core.skip_convs.parameters()
        yield from self.core.end_conv1.parameters()
        yield from self.core.end_conv2.parameters()


class PatchSTGAdapter(SALTBackbone):
    def __init__(self, nodes, input_steps=12, pred_steps=12, dropout=0.1,
                 adj=None, **_):
        super().__init__(input_steps, pred_steps)
        if adj is None:
            raise ValueError("PatchSTG requires adjacency")
        cls = _file_module(
            "official_patchstg", UPSTREAM / "PatchSTG" / "models" / "model.py"
        ).PatchSTG
        original, reordered, all_nodes, patch_size, patch_count = spectral_patch_indices(adj, 8)
        self.core = cls(
            tem_patchsize=3, tem_patchnum=self.total_steps // 3,
            node_num=nodes, spa_patchsize=patch_size, spa_patchnum=patch_count,
            tod=1, dow=1, layers=3, factors=1,
            input_dims=24, node_dims=16, tod_dims=8, dow_dims=8,
            ori_parts_idx=original, reo_parts_idx=reordered, reo_all_idx=all_nodes,
        )
        self.latent_dim = 56
        self.patch_len = 3
        # The official head reconstructs every input step. Our controlled
        # protocol exposes 24 context/mask steps but forecasts 12 future steps.
        self.core.regression_conv = nn.Conv2d(
            in_channels=(self.total_steps // self.patch_len) * self.latent_dim,
            out_channels=pred_steps, kernel_size=(1, 1), bias=True,
        )

    def _time(self, sequence):
        return torch.zeros((*sequence.shape[:3], 2), dtype=torch.long, device=sequence.device)

    def encode_sequence(self, sequence):
        embedded = self.core.embedding(sequence, self._time(sequence))
        patched = embedded[:, :, self.core.reo_all_idx, :]
        for block in self.core.spa_encoder:
            patched = block(patched)
        output = torch.zeros(
            patched.shape[0], patched.shape[1], self.core.node_num, patched.shape[-1],
            device=sequence.device, dtype=patched.dtype,
        )
        output[:, :, self.core.ori_parts_idx, :] = patched[:, :, self.core.reo_parts_idx, :]
        return output

    def forecast_from_sequence(self, sequence):
        hidden = self.encode_sequence(sequence)
        batch = hidden.shape[0]
        return self.core.regression_conv(
            hidden.transpose(2, 3).reshape(batch, -1, self.core.node_num, 1)
        )

    def align_teacher(self, teacher_hidden):
        steps = teacher_hidden.shape[1] // self.patch_len
        return teacher_hidden.view(
            teacher_hidden.shape[0], steps, self.patch_len,
            teacher_hidden.shape[2], teacher_hidden.shape[3],
        ).mean(dim=2)

    def encoder_parameters(self):
        yield self.future_token
        yield from _parameters_except(self.core, ("regression_conv",))

    def head_parameters(self):
        yield from self.core.regression_conv.parameters()


class FlashSTAdapter(SALTBackbone):
    def __init__(self, nodes, input_steps=12, pred_steps=12, dropout=0.1,
                 adj=None, train_x=None, **_):
        super().__init__(input_steps, pred_steps)
        if adj is None:
            raise ValueError("FlashST requires adjacency")
        prompt_cls = _file_module(
            "official_flashst_prompt", UPSTREAM / "FlashST" / "model" / "PromptNet.py"
        ).PromptNet
        embed = 16
        args = SimpleNamespace(
            mode="pretrain", node_dim=3 * embed, his=self.total_steps,
            embed_dim=embed, pred=pred_steps, num_layer=3,
            temp_dim_tid=0, temp_dim_diw=0, if_time_in_day=False,
            if_day_in_week=False, if_spatial=True, input_base_dim=1,
            input_extra_dim=0, data_type="traffic",
        )
        self.prompt = prompt_cls(args)
        self.register_buffer("lap_prompt", torch.from_numpy(laplacian_pe(adj, 3 * embed)))
        degree = np.maximum(adj.sum(1), 1e-8)
        self.register_buffer("norm_adj", torch.from_numpy((adj / degree[:, None]).astype(np.float32)))
        self.predictor = PDFormerAdapter(
            nodes, input_steps, pred_steps, dropout, adj, train_x,
            feature_dim=4 * embed,
        )
        self.latent_dim = self.predictor.latent_dim

    def _prompted(self, sequence):
        return self.prompt(
            sequence, sequence, nadj=self.norm_adj,
            lpls=self.lap_prompt, useGNN=True,
        )

    def encode_sequence(self, sequence):
        return self.predictor.encode_sequence(self._prompted(sequence))

    def forecast_from_sequence(self, sequence):
        return self.predictor.forecast_from_sequence(self._prompted(sequence))

    def encoder_parameters(self):
        yield self.future_token
        yield from self.prompt.parameters()
        for parameter in self.predictor.encoder_parameters():
            if parameter is not self.predictor.future_token:
                yield parameter

    def head_parameters(self):
        yield from self.predictor.head_parameters()


class TESTAMAdapter(SALTBackbone):
    def __init__(self, nodes, input_steps=12, pred_steps=12, dropout=0.1, **_):
        super().__init__(input_steps, pred_steps)
        cls = _file_module("official_testam", UPSTREAM / "TESTAM" / "model.py").TESTAM
        self.core = cls(
            num_nodes=nodes, in_dim=2, out_dim=1, hidden_size=32,
            layers=3, dropout=dropout, max_time_index=288,
        )
        self.latent_dim = 96

    def _input(self, sequence):
        batch, steps, nodes, _ = sequence.shape
        relative = torch.arange(steps, device=sequence.device, dtype=sequence.dtype)
        relative = relative.view(1, steps).expand(batch, -1) / 288.0
        features = torch.cat(
            [sequence, relative[:, :, None, None].expand(-1, -1, nodes, -1)], -1
        )
        return features.permute(0, 3, 2, 1)

    def encode_sequence(self, sequence):
        data = self._input(sequence)
        gate = self.core.gate_network
        n1, n2 = gate.We1 @ gate.memory, gate.We2 @ gate.memory
        supports = [
            torch.softmax(torch.relu(n1 @ n2.T), -1),
            torch.softmax(torch.relu(n2 @ n1.T), -1),
        ]
        time_index = data[:, -1, 0]
        current = ((time_index * self.core.max_time_index) % self.core.max_time_index).long()
        future = ((time_index * self.core.max_time_index + time_index.size(-1)) % self.core.max_time_index).long()
        _, identity = self.core.identity_expert(current, data[:, :-1].permute(0, 2, 3, 1))
        _, future_hidden = self.core.identity_expert(future)
        _, _, adaptive = self.core.adaptive_expert(data, future_hidden, supports)
        _, attention = self.core.attention_expert(data, future_hidden)
        hidden = torch.cat((identity[-1], adaptive[-1], attention), dim=-1)
        return hidden.permute(0, 2, 1, 3)

    def forecast_from_sequence(self, sequence):
        result = self.core(self._input(sequence))
        if isinstance(result, tuple):
            result = result[0]
        return result.permute(0, 3, 2, 1)[:, -self.pred_steps:]

    def encoder_parameters(self):
        yield self.future_token
        for name, parameter in self.core.named_parameters():
            if "gate_network" not in name and not name.endswith(".proj.weight") and not name.endswith(".proj.bias"):
                yield parameter

    def head_parameters(self):
        for name, parameter in self.core.named_parameters():
            if "gate_network" in name or name.endswith(".proj.weight") or name.endswith(".proj.bias"):
                yield parameter


class TSFormerAdapter(SALTBackbone):
    """Official TSFormer encoder adapted to 24-step city windows."""

    def __init__(self, nodes, input_steps=12, pred_steps=12, dropout=0.1, **_):
        super().__init__(input_steps, pred_steps)
        module = _import_from("TSFormer", UPSTREAM)
        self.core = module.TSFormer(
            patch_size=3, in_channel=1, embed_dim=128, num_heads=4,
            mlp_ratio=4, dropout=dropout, num_token=self.total_steps // 3,
            mask_ratio=0.75, encoder_depth=4, decoder_depth=1,
            mode="forecasting",
        )
        self.head = nn.Linear((self.total_steps // 3) * 128, pred_steps)
        self.latent_dim = 128
        self.patch_len = 3

    def encode_sequence(self, sequence):
        return self.core(sequence).permute(0, 2, 1, 3)

    def forecast_from_sequence(self, sequence):
        hidden = self.encode_sequence(sequence).permute(0, 2, 1, 3)
        output = self.head(hidden.flatten(-2))
        return output.transpose(1, 2).unsqueeze(-1)

    def align_teacher(self, teacher_hidden):
        steps = teacher_hidden.shape[1] // self.patch_len
        return teacher_hidden.view(
            teacher_hidden.shape[0], steps, self.patch_len,
            teacher_hidden.shape[2], teacher_hidden.shape[3],
        ).mean(dim=2)

    def encoder_parameters(self):
        yield self.future_token
        yield from self.core.parameters()

    def head_parameters(self):
        yield from self.head.parameters()


class STGormerAdapter(SALTBackbone):
    def __init__(self, nodes, input_steps=12, pred_steps=12, dropout=0.1,
                 adj=None, **_):
        super().__init__(input_steps, pred_steps)
        if adj is None:
            raise ValueError("STGormer requires adjacency")
        root = UPSTREAM / "STGormer"
        sys.path.insert(0, str(root))
        try:
            cls = importlib.import_module("model.models").MoESTar
        finally:
            sys.path.pop(0)
        args = SimpleNamespace(
            moe_status="SoftMoE", num_experts=6, moe_dropout=dropout,
            top_k=1, moe_add_ff=False, expertWeightsAda=False,
            expertWeights=[0.8, 0.2], pos_embed_T="timepos",
            num_timestamps=2016, tod_scaler=288, steps_per_day=288,
            cen_embed_S=False, attn_bias_S=False, num_shortpath=2,
            num_node_deg=max(nodes, 2), attn_mask_S=False,
            attn_mask_T=False, d_time_embed=24, d_space_embed=24,
            d_output=1, d_model=64, num_heads=4, mlp_ratio=4,
            layer_depth=2, dropout=dropout, layers=["S", "T"],
            moe_position="Full", dataset="METRLA", input_length=self.total_steps,
            output_length=pred_steps, mask_value_train=0.0,
            fft_status=False, num_nodes=nodes,
        )
        self.core = cls(args)
        self.register_buffer("graph", torch.from_numpy(np.asarray(adj, dtype=np.float32)))
        self.latent_dim = 64

    def _features(self, sequence):
        zeros = torch.zeros((*sequence.shape[:3], 2), device=sequence.device, dtype=sequence.dtype)
        return torch.cat((sequence, zeros), dim=-1)

    def encode_sequence(self, sequence):
        hidden, _ = self.core(self._features(sequence), self.graph)
        return hidden.permute(0, 2, 1, 3)

    def forecast_from_sequence(self, sequence):
        hidden = self.encode_sequence(sequence).permute(0, 2, 1, 3)
        batch, nodes, steps, dim = hidden.shape
        output = self.core.output_proj(hidden.reshape(batch, nodes, steps * dim))
        return output.view(batch, nodes, self.pred_steps, 1).transpose(1, 2)

    def encoder_parameters(self):
        yield self.future_token
        yield from self.core.encoder.parameters()

    def head_parameters(self):
        yield from self.core.output_proj.parameters()


BACKBONES = (
    "pdformer", "flashst", "patchstg", "testam",
    "patchtst", "staeformer", "stgormer", "tsformer",
)


def build_backbone(name: str, nodes: int, input_steps=12, pred_steps=12,
                   dropout=0.1, adj=None, train_x=None) -> SALTBackbone:
    classes = {
        "patchtst": PatchTSTAdapter,
        "staeformer": STAEformerAdapter,
        "pdformer": PDFormerAdapter,
        "flashst": FlashSTAdapter,
        "patchstg": PatchSTGAdapter,
        "testam": TESTAMAdapter,
        "tsformer": TSFormerAdapter,
        "stgormer": STGormerAdapter,
    }
    try:
        cls = classes[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown backbone {name!r}; choose from {BACKBONES}") from error
    return cls(nodes, input_steps, pred_steps, dropout, adj=adj, train_x=train_x)


def compatible_encoder_transfer(target: SALTBackbone, source_state: dict[str, torch.Tensor]):
    """Copy shared encoder tensors while rebuilding city-specific state and heads."""
    current = target.state_dict()
    copied, skipped = [], []
    head_fragments = (
        "core.head", "core.output_proj", "core.regression_conv",
        "core.skip_convs", "core.end_conv", ".head.", "head.",
    )
    city_fragments = (
        "adaptive_embedding", "sensor_embedding", "node_emb", "node_features",
        "pattern_keys", "geo_mask", "sem_mask", "lap", "graph",
        "norm_adj", "lap_prompt",
    )
    for name, value in source_state.items():
        if any(fragment in name for fragment in head_fragments + city_fragments):
            skipped.append(name)
        elif name in current and current[name].shape == value.shape:
            current[name] = value
            copied.append(name)
        else:
            skipped.append(name)
    target.load_state_dict(current)
    if not copied:
        raise ValueError("No compatible encoder tensors were transferred")
    return {"copied": copied, "skipped": skipped}
