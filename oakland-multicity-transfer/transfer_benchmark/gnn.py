from __future__ import annotations

import torch
from torch import nn


class _GraphGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(2 * (input_dim + hidden_dim), 2 * hidden_dim)
        self.candidate = nn.Linear(2 * (input_dim + hidden_dim), hidden_dim)

    def forward(self, x, hidden, support):
        x_graph = torch.einsum("nm,bmd->bnd", support, x)
        h_graph = torch.einsum("nm,bmd->bnd", support, hidden)
        joined = torch.cat((x, x_graph, hidden, h_graph), dim=-1)
        reset, update = torch.sigmoid(self.gates(joined)).chunk(2, dim=-1)
        candidate_in = torch.cat((x, x_graph, reset * hidden, reset * h_graph), dim=-1)
        candidate = torch.tanh(self.candidate(candidate_in))
        return update * hidden + (1.0 - update) * candidate


class TrafficGNN(nn.Module):
    """Small transferable recurrent traffic GNN with a common benchmark contract."""

    def __init__(self, nodes: int, input_steps: int, pred_steps: int,
                 adjacency, kind: str, hidden_dim: int = 64,
                 node_embedding_dim: int = 16):
        super().__init__()
        if kind not in {"dcrnn", "agcrn"}:
            raise ValueError(kind)
        self.kind = kind
        self.nodes = nodes
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.hidden_dim = hidden_dim
        base = torch.as_tensor(adjacency, dtype=torch.float32)
        base = base + torch.eye(nodes)
        base = base / base.sum(-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("fixed_support", base)
        if kind == "agcrn":
            self.node_embeddings = nn.Parameter(torch.randn(nodes, node_embedding_dim) * 0.1)
        else:
            self.register_parameter("node_embeddings", None)
        self.cell = _GraphGRUCell(1, hidden_dim)
        self.head = nn.Linear(hidden_dim, pred_steps)

    def support(self):
        if self.kind == "dcrnn":
            return self.fixed_support
        logits = torch.relu(self.node_embeddings @ self.node_embeddings.T)
        return torch.softmax(logits, dim=-1)

    def encode(self, history):
        # history: [B,T,N,1]
        hidden = history.new_zeros(history.shape[0], history.shape[2], self.hidden_dim)
        support = self.support()
        for step in range(history.shape[1]):
            hidden = self.cell(history[:, step], hidden, support)
        return hidden

    def forward(self, history):
        prediction = self.head(self.encode(history))
        return prediction.transpose(1, 2).unsqueeze(-1)

    def encoder_parameters(self):
        yield from self.cell.parameters()
        if self.node_embeddings is not None:
            yield self.node_embeddings

    def head_parameters(self):
        yield from self.head.parameters()


def transferable_state(target: TrafficGNN, source_state):
    current = target.state_dict()
    copied, skipped = [], []
    for name, value in source_state.items():
        city_specific = name in {"node_embeddings", "fixed_support"}
        forecast_head = name.startswith("head.")
        if not city_specific and not forecast_head and name in current and current[name].shape == value.shape:
            current[name] = value
            copied.append(name)
        else:
            skipped.append(name)
    target.load_state_dict(current)
    if not copied:
        raise ValueError("No transferable GNN encoder tensors were found")
    return {"copied": copied, "skipped": skipped}
