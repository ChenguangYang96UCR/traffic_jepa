from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from types import ModuleType
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"


def _isolated_import(module_name: str, root: Path, top_package: str):
    """Import an upstream repo without leaking its generic package names.

    Several vendored traffic repositories call their top-level package either
    ``model`` or ``libcity``.  Temporarily isolating that namespace prevents one
    official implementation from shadowing another in the unified benchmark.
    """
    prefix = top_package + "."
    saved = {
        name: module for name, module in tuple(sys.modules.items())
        if name == top_package or name.startswith(prefix)
    }
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.pop(0)
        for name in tuple(sys.modules):
            if name == top_package or name.startswith(prefix):
                del sys.modules[name]
        sys.modules.update(saved)


def _load_sdgformer_module():
    """Load the released model file without requiring a LibCity install.

    The official repository vendors a partial LibCity tree and its model file
    only needs ``AbstractTrafficStateModel`` during construction.  Supplying
    that small interface locally avoids both a global ``libcity`` dependency
    and collisions with the separately vendored PDFormer LibCity package.
    """
    source = (
        UPSTREAM / "SDGFormer" / "libcity" / "model" /
        "traffic_flow_prediction" / "SDGFormer.py"
    )
    if not source.exists():
        raise FileNotFoundError(
            f"Missing vendored SDGFormer source: {source}. "
            "Copy the package's upstream/SDGFormer directory; do not run "
            "`pip install libcity`."
        )

    class AbstractTrafficStateModel(nn.Module):
        def __init__(self, config, data_feature):
            super().__init__()
            self.data_feature = data_feature

    package = ModuleType("libcity")
    model_package = ModuleType("libcity.model")
    loss_module = ModuleType("libcity.model.loss")
    abstract_module = ModuleType("libcity.model.abstract_traffic_state_model")
    abstract_module.AbstractTrafficStateModel = AbstractTrafficStateModel
    model_package.loss = loss_module
    package.model = model_package

    names = (
        "libcity", "libcity.model", "libcity.model.loss",
        "libcity.model.abstract_traffic_state_model",
    )
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    sys.modules.update({
        "libcity": package,
        "libcity.model": model_package,
        "libcity.model.loss": loss_module,
        "libcity.model.abstract_traffic_state_model": abstract_module,
    })
    try:
        spec = importlib.util.spec_from_file_location(
            "_traffic_benchmark_official_sdgformer", source
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def _laplacian_pe(adjacency: np.ndarray, dim: int = 8) -> np.ndarray:
    adjacency = np.maximum(adjacency, adjacency.T).astype(np.float64)
    adjacency += np.eye(adjacency.shape[0])
    degree = np.maximum(adjacency.sum(1), 1e-8)
    normalized = adjacency / np.sqrt(degree[:, None] * degree[None, :])
    _, vectors = np.linalg.eigh(np.eye(adjacency.shape[0]) - normalized)
    available = vectors[:, 1:1 + min(dim, max(vectors.shape[1] - 1, 0))]
    if available.shape[1] < dim:
        available = np.pad(available, ((0, 0), (0, dim - available.shape[1])))
    return available.astype(np.float32)


def _except_prefixes(module: nn.Module, prefixes: tuple[str, ...]):
    for name, parameter in module.named_parameters():
        if not name.startswith(prefixes):
            yield parameter


class SDGFormerForecast(nn.Module):
    """Official SDGFormer core under the benchmark's traffic-only contract."""

    def __init__(self, nodes: int, input_steps: int, pred_steps: int,
                 adjacency: np.ndarray, train_samples: int, dropout: float = 0.1):
        super().__init__()
        if adjacency.shape != (nodes, nodes):
            raise ValueError(f"SDGFormer adjacency is {adjacency.shape}, expected {(nodes, nodes)}")
        module = _load_sdgformer_module()
        config = {
            "dataset": "released_city_windows", "device": torch.device("cpu"),
            "input_window": input_steps, "output_window": pred_steps,
            "output_dim": 1, "embed_dim": 64, "skip_dim": 256,
            "lape_dim": 8, "mlp_ratio": 4, "qkv_bias": True,
            "drop": dropout, "attn_drop": 0.0, "drop_path": 0.3,
            "enc_depth": 6, "type_ln": "post", "graph": 1,
            "add_time_in_day": False, "add_day_in_week": False,
            "use_curriculum_learning": False, "step_size": 1,
            "max_epoch": 100,
        }
        features = {
            "scaler": None, "num_nodes": nodes, "feature_dim": 1,
            "ext_dim": 0, "num_batches": max(1, math.ceil(train_samples / 16)),
        }
        self.core = module.SDGFormer(config, features)
        self.representation_dim = 64
        self.register_buffer("laplacian", torch.from_numpy(_laplacian_pe(adjacency, 8)))

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        hidden = self.core.enc_embed_layer(history, self.laplacian)
        for block in self.core.encoder_blocks:
            hidden = block(hidden, 1, 1, None, None)
        return hidden

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.core({"X": history}, self.laplacian)

    def encoder_parameters(self):
        yield from _except_prefixes(
            self.core, ("skip_convs.", "end_conv1.", "end_conv2.")
        )

    def head_parameters(self):
        yield from self.core.skip_convs.parameters()
        yield from self.core.end_conv1.parameters()
        yield from self.core.end_conv2.parameters()


class DSTFGCNForecast(nn.Module):
    """Official DSTFGCN core adapted to windows without absolute time IDs.

    The released city samples contain relative 12-step windows but no verified
    absolute timestamp.  ``N_t=1`` therefore disables unverifiable calendar
    lookup while preserving DSTFGCN's learned relative temporal graph.
    """

    def __init__(self, nodes: int, input_steps: int, pred_steps: int,
                 adjacency: np.ndarray, dropout: float = 0.2):
        super().__init__()
        self.pred_steps = pred_steps
        self.representation_dim = 48
        if input_steps != 12:
            raise ValueError("The official DSTFGCN implementation requires 12 history steps")
        if adjacency.shape != (nodes, nodes):
            raise ValueError(f"DSTFGCN adjacency is {adjacency.shape}, expected {(nodes, nodes)}")
        module = _isolated_import(
            "model.DSTFGCN", UPSTREAM / "DSTFGCN", "model"
        )
        config = {
            "drop": dropout, "layers": 6, "num_nodes": nodes,
            "hidden_dimension": 48, "embedding_dimension": 10,
            "N_t": 1, "temporal_dim": [10, 10, 4, 4],
            # The released head is structurally tied to the 12 input steps.
            # Train its native 12-step output and select the requested prefix.
            "input_dim": 1, "output_dim": 12,
        }
        self.core = module.DSTFGCN(
            torch.device("cpu"), config, np.asarray(adjacency, dtype=np.float32),
            skip_channels=48 * 8,
        )
        # The release stores these tensors as ordinary attributes, which keeps
        # them on CPU after model.to(cuda). Register them as proper buffers.
        for submodule in self.core.modules():
            if hasattr(submodule, "adj_one"):
                value = submodule.adj_one
                delattr(submodule, "adj_one")
                submodule.register_buffer("adj_one", value)
        adjacency_tensor = self.core.adj
        delattr(self.core, "adj")
        self.core.register_buffer("adj", adjacency_tensor)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """Return the released encoder's final spatiotemporal state [B,N,D]."""
        time_id = torch.zeros(history.shape[0], dtype=torch.long, device=history.device)
        x = self.core.start_conv(history.permute(0, 3, 2, 1))
        for index in range(self.core.layers):
            residual = x
            shared = self.core.ST_block(x, time_id)
            individual = self.core.layer_ST_block[index](x, time_id)
            x = torch.relu(torch.cat((shared, individual), dim=1))
            x = self.core.result_fuse[index](x)
            x = self.core.bn[index](x + residual[:, :, :, -x.size(3):])
        return x.mean(dim=-1).transpose(1, 2)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        time_id = torch.zeros(history.shape[0], dtype=torch.long, device=history.device)
        return self.core(history.permute(0, 3, 2, 1), time_id)[:, :self.pred_steps]

    def encoder_parameters(self):
        yield from _except_prefixes(
            self.core, ("skip_convs.", "end_conv_1.", "end_conv_2.")
        )

    def head_parameters(self):
        yield from self.core.skip_convs.parameters()
        yield from self.core.end_conv_1.parameters()
        yield from self.core.end_conv_2.parameters()


def recent_encoder_transfer(target: nn.Module, source_state: dict[str, torch.Tensor]):
    """Transfer portable encoder state and rebuild graph/node/head tensors."""
    current = target.state_dict()
    copied, skipped = [], []
    head_fragments = (
        "skip_convs.", "end_conv1.", "end_conv2.",
        "end_conv_1.", "end_conv_2.",
    )
    city_fragments = ("node_embeddings", "nodenum", "laplacian", "adj_one", ".adj")
    for name, value in source_state.items():
        incompatible = any(fragment in name for fragment in head_fragments + city_fragments)
        if not incompatible and name in current and current[name].shape == value.shape:
            current[name] = value
            copied.append(name)
        else:
            skipped.append(name)
    target.load_state_dict(current)
    if not copied:
        raise ValueError("No transferable SDGFormer/DSTFGCN encoder tensors were found")
    return {"copied": copied, "skipped": skipped}
