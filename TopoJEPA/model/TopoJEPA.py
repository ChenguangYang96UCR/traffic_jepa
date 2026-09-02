import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import copy
import os


class SimpleGraphEncoder(nn.Module):

    def __init__(self, seq_len, d_model, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, normalized_adjacency):
        # x: B,L,N -> B,N,L
        hidden = self.input_projection(x.transpose(1, 2))
        for layer in self.layers:
            message = torch.matmul(normalized_adjacency, hidden)
            hidden = hidden + self.dropout(F.gelu(layer(message)))
        return hidden


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.model_variant = getattr(configs, 'model_variant', 'original')
        self.jepa_weight = getattr(configs, 'jepa_weight', 0.0)
        self.topo_weight = getattr(configs, 'topo_weight', 0.0)
        self.text_weight = getattr(configs, 'text_weight', 0.0)
        self.text_embed_dim = getattr(configs, 'text_embed_dim', 512)
        self.use_gnn = bool(getattr(configs, 'use_gnn', False))
        self.alignment_weight = getattr(configs, 'alignment_weight', 0.0)
        if self.alignment_weight > 0 and not self.use_gnn:
            raise ValueError('alignment_weight > 0 requires use_gnn=True')
        self.ema_momentum = getattr(configs, 'ema_momentum', 0.996)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if self.use_gnn:
            adjacency = self._load_adjacency(getattr(configs, 'adj_path', ''))
            self.register_buffer('normalized_adjacency', adjacency)
            self.graph_encoder = SimpleGraphEncoder(
                configs.seq_len, configs.d_model,
                num_layers=getattr(configs, 'gnn_layers', 2),
                dropout=getattr(configs, 'gnn_dropout', configs.dropout))
            self.fusion_gate = nn.Linear(2 * configs.d_model, configs.d_model)
        if self.model_variant in ('predictor', 'jepa'):
            self.predictor = nn.Sequential(
                nn.LayerNorm(configs.d_model),
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, configs.d_model)
            )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)


        if self.model_variant == 'jepa':
            self.target_embedding = copy.deepcopy(self.enc_embedding)
            self.target_encoder = copy.deepcopy(self.encoder)
            for module in (self.target_embedding, self.target_encoder):
                module.requires_grad_(False)
            if self.use_gnn:
                self.target_graph_encoder = copy.deepcopy(self.graph_encoder)
                self.target_fusion_gate = copy.deepcopy(self.fusion_gate)
                for module in (self.target_graph_encoder, self.target_fusion_gate):
                    module.requires_grad_(False)
            self.text_predictor = nn.Sequential(
                nn.LayerNorm(configs.d_model),
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, self.text_embed_dim)
            )

    @staticmethod
    def _load_adjacency(path):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f'use_gnn=True requires a valid adj_path: {path}')
        if path.endswith('.npy'):
            adjacency = np.load(path)
        else:
            raw = np.genfromtxt(path, delimiter=',', skip_header=1)
            adjacency = raw[:, 1:] if raw.shape[1] == raw.shape[0] + 1 else raw
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f'Adjacency must be square, got {adjacency.shape}: {path}')
        adjacency = np.nan_to_num(adjacency, nan=0.0)
        adjacency = np.maximum(adjacency, adjacency.T)
        adjacency = adjacency + np.eye(adjacency.shape[0], dtype=adjacency.dtype)
        degree = adjacency.sum(axis=1)
        inv_sqrt = np.power(np.maximum(degree, 1e-12), -0.5)
        normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
        return torch.tensor(normalized, dtype=torch.float32)

    def _fuse_graph(self, transformer_rep, x, target=False):
        if not self.use_gnn:
            return transformer_rep, None, transformer_rep[:, :x.shape[-1], :]
        num_variables = x.shape[-1]
        if num_variables != self.normalized_adjacency.shape[0]:
            raise ValueError(
                f'Input has {num_variables} variables but adjacency has '
                f'{self.normalized_adjacency.shape[0]} nodes')
        graph_encoder = self.target_graph_encoder if target else self.graph_encoder
        fusion_gate = self.target_fusion_gate if target else self.fusion_gate
        if target:
            graph_encoder.eval()
            fusion_gate.eval()
        transformer_variables = transformer_rep[:, :num_variables, :]
        graph_variables = graph_encoder(x, self.normalized_adjacency)
        gate = torch.sigmoid(fusion_gate(torch.cat(
            [transformer_variables, graph_variables], dim=-1)))
        fused_variables = gate * transformer_variables + (1.0 - gate) * graph_variables
        fused = torch.cat([fused_variables, transformer_rep[:, num_variables:, :]], dim=1)
        return fused, graph_variables, transformer_variables

    def _encode(self, x, x_mark, target=False):
        embedding = self.target_embedding if target else self.enc_embedding
        encoder = self.target_encoder if target else self.encoder
        if target:
            embedding.eval()
            encoder.eval()
        representation = embedding(x, x_mark)
        representation, attns = encoder(representation, attn_mask=None)
        return representation, attns

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update for the JEPA target embedding and encoder."""
        if self.model_variant != 'jepa' or (self.jepa_weight <= 0 and
                                            self.topo_weight <= 0 and
                                            self.text_weight <= 0 and
                                            self.alignment_weight <= 0):
            return
        for online, target in ((self.enc_embedding, self.target_embedding),
                               (self.encoder, self.target_encoder)):
            for online_param, target_param in zip(online.parameters(), target.parameters()):
                target_param.data.mul_(self.ema_momentum).add_(
                    online_param.data, alpha=1.0 - self.ema_momentum)
        if self.use_gnn:
            for online, target in ((self.graph_encoder, self.target_graph_encoder),
                                   (self.fusion_gate, self.target_fusion_gate)):
                for online_param, target_param in zip(online.parameters(), target.parameters()):
                    target_param.data.mul_(self.ema_momentum).add_(
                        online_param.data, alpha=1.0 - self.ema_momentum)

    def forward_with_jepa(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                          target_x, target_mark, target_text=None):
        if self.model_variant != 'jepa' or (self.jepa_weight <= 0 and
                                            self.topo_weight <= 0 and
                                            self.text_weight <= 0 and
                                            self.alignment_weight <= 0):
            zero = x_enc.new_zeros(())
            return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec), zero, zero, zero, zero

        online_input, means, stdev = self._normalize_with_stats(x_enc)
        num_variables = x_enc.shape[-1]
        online_context, attns = self._encode(online_input, x_mark_enc)
        online_context, graph_variables, transformer_variables = self._fuse_graph(
            online_context, online_input, target=False)
        predicted_rep = self.predictor(online_context)
        forecast = self._project_representation(
            predicted_rep, num_variables, means, stdev)

        target_input = self._normalized_view(target_x)
        with torch.no_grad():
            target_rep, _ = self._encode(target_input, target_mark, target=True)
            target_rep, _, _ = self._fuse_graph(target_rep, target_input, target=True)

        predicted_variables = predicted_rep[:, :num_variables, :]
        target_variables = target_rep[:, :num_variables, :]

        jepa_loss = F.mse_loss(
            F.layer_norm(predicted_variables, predicted_variables.shape[-1:]),
            F.layer_norm(target_variables, target_variables.shape[-1:]))
        topo_loss = self._topology_wasserstein_loss(
            predicted_variables, target_variables)
        text_loss = x_enc.new_zeros(())
        if self.text_weight > 0:
            if target_text is None:
                raise ValueError('text_weight > 0 requires cached CLIP embeddings')
            predicted_text = F.normalize(
                self.text_predictor(online_context[:, :num_variables, :]), dim=-1)
            target_text = F.normalize(target_text.detach(), dim=-1)
            text_loss = 1.0 - (predicted_text * target_text).sum(dim=-1).mean()
        alignment_loss = x_enc.new_zeros(())
        if self.use_gnn and self.alignment_weight > 0:
            alignment_loss = self._cramer_alignment_loss(
                graph_variables, transformer_variables)
        return forecast, jepa_loss, topo_loss, text_loss, alignment_loss

    @staticmethod
    def _cramer_alignment_loss(graph_rep, transformer_rep):
        graph_rep = F.normalize(graph_rep, dim=-1)
        transformer_rep = F.normalize(transformer_rep, dim=-1)
        cross = torch.cdist(graph_rep, transformer_rep, p=2).mean()
        graph_spread = torch.cdist(graph_rep, graph_rep, p=2).mean()
        transformer_spread = torch.cdist(
            transformer_rep, transformer_rep, p=2).mean()
        return (2.0 * cross - graph_spread - transformer_spread).clamp_min(0.0)

    @staticmethod
    def _h0_persistence_deaths(points):
        batch_size, num_points, _ = points.shape
        if num_points < 2:
            return points.new_zeros((batch_size, 0))
        distances = torch.cdist(points, points, p=2)
        upper = torch.triu_indices(num_points, num_points, offset=1,
                                   device=points.device)
        diagrams = []
        for batch_index in range(batch_size):
            edge_weights = distances[batch_index, upper[0], upper[1]]
            order = torch.argsort(edge_weights.detach())
            parent = list(range(num_points))

            def find(node):
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            selected = []
            for edge_index in order.tolist():
                left = find(int(upper[0, edge_index]))
                right = find(int(upper[1, edge_index]))
                if left != right:
                    parent[left] = right
                    selected.append(edge_weights[edge_index])
                    if len(selected) == num_points - 1:
                        break
            diagrams.append(torch.sort(torch.stack(selected)).values)
        return torch.stack(diagrams)

    def _topology_wasserstein_loss(self, context_rep, target_rep):
        context_points = F.normalize(context_rep, dim=-1)
        target_points = F.normalize(target_rep, dim=-1)
        context_deaths = self._h0_persistence_deaths(context_points)
        target_deaths = self._h0_persistence_deaths(target_points)
        if context_deaths.shape[-1] == 0:
            return context_rep.new_zeros(())
        return torch.mean((context_deaths - target_deaths) ** 2)

    def _normalized_view(self, x):
        normalized, _, _ = self._normalize_with_stats(x)
        return normalized

    def _normalize_with_stats(self, x):
        if not self.use_norm:
            return x, None, None
        means = x.mean(1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True,
                                     unbiased=False) + 1e-5)
        return centered / stdev, means, stdev

    def _project_representation(self, representation, num_variables,
                                means=None, stdev=None):
        output = self.projector(representation).permute(0, 2, 1)
        output = output[:, :, :num_variables]
        if self.use_norm:
            output = output * stdev[:, 0, :].unsqueeze(1)
            output = output + means[:, 0, :].unsqueeze(1)
        return output

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, means, stdev = self._normalize_with_stats(x_enc)
        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out, attns = self._encode(x_enc, x_mark_enc)
        enc_out, _, _ = self._fuse_graph(enc_out, x_enc, target=False)
        if self.model_variant in ('predictor', 'jepa'):
            enc_out = self.predictor(enc_out)

        # B N E -> B N S -> B S N
        dec_out = self._project_representation(enc_out, N, means, stdev)

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
