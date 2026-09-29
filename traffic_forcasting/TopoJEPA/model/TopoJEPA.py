import torch
import torch.nn as nn
import torch.nn.functional as F
from traffic_forcasting.TopoJEPA.layers.Transformer_EncDec import Encoder, EncoderLayer
from traffic_forcasting.TopoJEPA.layers.SelfAttention_Family import FullAttention, AttentionLayer
from traffic_forcasting.TopoJEPA.layers.Embed import DataEmbedding_inverted
import numpy as np
import copy
import os
import math


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


class IncidentContextFusion(nn.Module):
    """IGSTGNN-style node-specific incident conditioning for JEPA tokens."""

    def __init__(self, hidden_dim, num_descriptions=2048, num_types=64,
                 num_positions=12, dropout=0.1):
        super().__init__()
        self.position_embedding = nn.Embedding(num_positions, 8)
        self.description_embedding = nn.Embedding(num_descriptions, 32)
        self.type_embedding = nn.Embedding(num_types, 8)
        self.holiday_embedding = nn.Embedding(2, 4)
        self.incident_fusion = nn.Sequential(
            nn.Linear(8 + 32 + 8 + 4, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.distance_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _index(values, embedding, name):
        values = values.long()
        if torch.any(values < 0) or torch.any(values >= embedding.num_embeddings):
            lower = int(values.min().item())
            upper = int(values.max().item())
            raise ValueError(
                f'{name} indices [{lower}, {upper}] exceed embedding range '
                f'[0, {embedding.num_embeddings - 1}]. Adjust the matching '
                'incident cardinality argument.')
        return values

    def forward(self, variable_tokens, incident_data):
        features = incident_data['features']
        positions = incident_data['position']
        distances = incident_data['distances']
        if features.ndim != 2 or features.shape[-1] != 4:
            raise ValueError(
                f'incident features must be [batch,4], got {features.shape}')
        if distances.ndim != 3 or distances.shape[1] != variable_tokens.shape[1] or distances.shape[2] != 3:
            raise ValueError(
                'incident distances must be [batch,node,3] and match the '
                f'JEPA variables; got {distances.shape} for '
                f'{variable_tokens.shape[1]} variables')

        # Match the released IGSTGNN implementation: the first element
        # (Incident Time) is loaded but the categorical embedding uses
        # Description, Type, Holiday, plus incident_position.
        descriptions = self._index(
            features[:, 1], self.description_embedding, 'Description')
        types = self._index(features[:, 2], self.type_embedding, 'Type')
        holidays = self._index(
            features[:, 3], self.holiday_embedding, 'Holiday')
        positions = self._index(
            positions.reshape(-1), self.position_embedding,
            'incident_position')
        incident_embedding = self.incident_fusion(torch.cat([
            self.position_embedding(positions),
            self.description_embedding(descriptions),
            self.type_embedding(types),
            self.holiday_embedding(holidays),
        ], dim=-1))

        distance_mask = distances.abs().sum(dim=-1, keepdim=True) > 0
        # A cropped-city sample may have no sensor spatially associated with
        # its incident. IGSTGNN treats that as zero incident context instead
        # of rejecting the sample.
        distance_context = F.softmax(
            self.distance_encoder(distances), dim=1)
        keys = self.key(incident_embedding).unsqueeze(1)
        values = self.value(incident_embedding).unsqueeze(1).expand_as(
            variable_tokens)
        semantic_logits = (
            self.query(variable_tokens) * keys).sum(dim=-1, keepdim=True)
        semantic_logits = semantic_logits / math.sqrt(variable_tokens.shape[-1])
        semantic_attention = F.softmax(
            semantic_logits.masked_fill(~distance_mask, -1e7), dim=1)
        semantic_attention = semantic_attention * distance_mask
        fused_weights = self.fusion(torch.cat([
            semantic_attention.expand_as(variable_tokens), distance_context
        ], dim=-1))
        fused_weights = F.softmax(
            fused_weights.masked_fill(~distance_mask, -1e7), dim=1)
        fused_weights = fused_weights * distance_mask
        incident_context = fused_weights * values
        return self.norm(variable_tokens + incident_context), incident_context


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
        self.use_incident = bool(getattr(configs, 'incident', False))
        self.use_forecast_mask = bool(getattr(configs, 'forecast_mask', False))
        self.forecast_mask_floor = float(
            getattr(configs, 'forecast_mask_floor', 0.1))
        self.pretrain_objective = getattr(
            configs, 'pretrain_objective', 'forecast')
        self.use_reconstruction = self.pretrain_objective in (
            'reconstruction', 'forecast_reconstruction')
        self.incident_scale = float(getattr(configs, 'incident_scale', 1.0))
        self.incident_sigma = float(getattr(configs, 'incident_sigma', 1.0))
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
        if self.use_reconstruction:
            # A deliberately light decoder forces the online encoder, rather
            # than a powerful reconstruction network, to learn the missing
            # traffic pattern. It is discarded/frozen for downstream use.
            self.reconstruction_head = nn.Linear(
                configs.d_model, configs.seq_len, bias=True)
        if self.use_forecast_mask:
            mask_hidden = max(configs.d_model // 4, 8)
            self.forecast_mask_head = nn.Sequential(
                nn.LayerNorm(configs.d_model),
                nn.Linear(configs.d_model, mask_hidden),
                nn.GELU(),
                nn.Linear(mask_hidden, configs.pred_len),
            )
        if self.use_incident:
            if self.incident_sigma <= 0:
                raise ValueError('--incident_sigma must be positive')
            self.incident_fusion = IncidentContextFusion(
                configs.d_model,
                num_descriptions=getattr(
                    configs, 'incident_num_descriptions', 2048),
                num_types=getattr(configs, 'incident_num_types', 64),
                num_positions=getattr(configs, 'incident_num_positions', 12),
                dropout=configs.dropout)
            self.incident_output = nn.Linear(configs.d_model, 1, bias=False)


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

    @torch.no_grad()
    def copy_target_encoder_to_online(self):
        """Initialize the deployable encoder from the EMA pretraining teacher."""
        if self.model_variant != 'jepa':
            raise ValueError('EMA transfer requires model_variant=jepa')
        self.enc_embedding.load_state_dict(self.target_embedding.state_dict())
        self.encoder.load_state_dict(self.target_encoder.state_dict())
        if self.use_gnn:
            self.graph_encoder.load_state_dict(
                self.target_graph_encoder.state_dict())
            self.fusion_gate.load_state_dict(
                self.target_fusion_gate.state_dict())

    def reset_forecast_head(self):
        """Reset the supervised head so it cannot inherit forecast training."""
        nn.init.xavier_uniform_(self.projector.weight)
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)
        if self.use_incident:
            nn.init.xavier_uniform_(self.incident_output.weight)

    def configure_pretraining(self):
        """Configure the selected auxiliary objective while always using JEPA."""
        if self.model_variant != 'jepa':
            raise ValueError('JEPA pretraining requires model_variant=jepa')
        self.requires_grad_(True)
        self.text_predictor.requires_grad_(self.text_weight > 0)
        uses_forecast = self.pretrain_objective in (
            'forecast', 'masked_forecast', 'forecast_reconstruction')
        self.projector.requires_grad_(uses_forecast)
        if self.use_reconstruction:
            self.reconstruction_head.requires_grad_(True)
        if self.use_incident:
            self.incident_output.requires_grad_(True)
        for module in (self.target_embedding, self.target_encoder):
            module.requires_grad_(False)
        if self.use_gnn:
            for module in (self.target_graph_encoder,
                           self.target_fusion_gate):
                module.requires_grad_(False)

    def configure_lora(self, rank=8, alpha=16.0, dropout=0.0):
        """Adapt only online attention Q/V and the supervised forecast head."""
        from traffic_forcasting.TopoJEPA.layers.lora import LoRALinear
        self.requires_grad_(False)
        for layer in self.encoder.attn_layers:
            for name in ('query_projection', 'value_projection'):
                base = getattr(layer.attention, name)
                if isinstance(base, LoRALinear):
                    raise ValueError('LoRA has already been installed')
                setattr(layer.attention, name, LoRALinear(base, rank, alpha, dropout))
        self.projector.requires_grad_(True)
        if self.use_incident:
            self.incident_output.requires_grad_(True)
        self._lora_enabled = True

    def configure_finetuning(self, strategy='full', unfreeze_layers=1):
        """Select trainable online parameters for full or partial fine-tuning."""
        if strategy not in ('full', 'partial', 'frozen'):
            raise ValueError(f'Unknown finetune strategy: {strategy}')
        if unfreeze_layers < 0:
            raise ValueError('unfreeze_layers must be non-negative')

        for parameter in self.parameters():
            parameter.requires_grad = strategy == 'full'

        # EMA modules are never optimized during downstream supervised training.
        for module in (self.target_embedding, self.target_encoder):
            module.requires_grad_(False)
        if self.use_gnn:
            for module in (self.target_graph_encoder,
                           self.target_fusion_gate):
                module.requires_grad_(False)
        if self.use_forecast_mask:
            # Auxiliary loss controller: retained in checkpoints for strict
            # loading, but never updated during downstream fine-tuning.
            self.forecast_mask_head.requires_grad_(False)
        if self.use_reconstruction:
            self.reconstruction_head.requires_grad_(False)

        if strategy in ('partial', 'frozen'):
            # Start frozen, then expose the task head, latent predictor and the
            # final N Transformer blocks.  With N=0 this is a head/predictor probe.
            for parameter in self.parameters():
                parameter.requires_grad = False
            self.projector.requires_grad_(True)
            if self.use_incident:
                self.incident_output.requires_grad_(True)
            if strategy == 'frozen':
                return

            self.predictor.requires_grad_(True)
            if self.use_incident:
                self.incident_fusion.requires_grad_(True)
            if self.use_gnn:
                self.graph_encoder.requires_grad_(True)
                self.fusion_gate.requires_grad_(True)
            layers = self.encoder.attn_layers
            for layer in layers[max(0, len(layers) - unfreeze_layers):]:
                layer.requires_grad_(True)
            if unfreeze_layers > 0 and self.encoder.norm is not None:
                self.encoder.norm.requires_grad_(True)
        self.text_predictor.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, '_lora_enabled', False):
            # Keep pretrained feature extraction deterministic, while allowing
            # explicitly configured adapter dropout during LoRA training.
            from traffic_forcasting.TopoJEPA.layers.lora import LoRALinear
            self.encoder.eval()
            for module in self.encoder.modules():
                if isinstance(module, LoRALinear):
                    module.dropout.train(mode)
        # Frozen feature extractors must also disable dropout during adaptation.
        for module in self.modules():
            if module is not self:
                parameters = list(module.parameters())
                if parameters and not any(p.requires_grad for p in parameters):
                    module.eval()
        return self

    def _fuse_incident(self, representation, num_variables, incident_data):
        if not self.use_incident:
            return representation, None
        if incident_data is None:
            raise ValueError('--incident requires incident data in every batch')
        variables = representation[:, :num_variables, :]
        variables, incident_context = self.incident_fusion(
            variables, incident_data)
        representation = torch.cat(
            [variables, representation[:, num_variables:, :]], dim=1)
        return representation, incident_context

    def _incident_decay(self, incident_context, pred_len):
        if incident_context is None:
            return None
        steps = torch.arange(
            1, pred_len + 1, dtype=incident_context.dtype,
            device=incident_context.device)
        decay = torch.exp(
            -(steps ** 2) / (2.0 * self.incident_sigma ** 2))
        node_effect = self.incident_output(incident_context).squeeze(-1)
        return self.incident_scale * decay.view(1, pred_len, 1) * node_effect.unsqueeze(1)

    def forward_with_jepa(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                          target_x, target_mark, target_text=None,
                          incident_data=None, return_forecast_mask=False,
                          input_mask=None, return_pretrain_aux=False):
        if self.model_variant != 'jepa' or (self.jepa_weight <= 0 and
                                            self.topo_weight <= 0 and
                                            self.text_weight <= 0 and
                                            self.alignment_weight <= 0):
            zero = x_enc.new_zeros(())
            result = (self.forward(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                incident_data=incident_data), zero, zero, zero, zero)
            if return_pretrain_aux:
                return (*result, {
                    'forecast_mask': None,
                    'reconstruction': None,
                })
            return (*result, None) if return_forecast_mask else result

        online_input, means, stdev = self._normalize_with_stats(x_enc)
        if input_mask is not None:
            if input_mask.shape != online_input.shape:
                raise ValueError(
                    f'Input mask {input_mask.shape} does not match input '
                    f'{online_input.shape}')
            # Zero is the per-sample mean after normalization, avoiding a
            # special raw-value token whose meaning changes across cities.
            online_input = online_input.masked_fill(input_mask.bool(), 0.0)
        num_variables = x_enc.shape[-1]
        online_context, attns = self._encode(online_input, x_mark_enc)
        online_context, graph_variables, transformer_variables = self._fuse_graph(
            online_context, online_input, target=False)
        online_context, incident_context = self._fuse_incident(
            online_context, num_variables, incident_data)
        predicted_rep = self.predictor(online_context)
        forecast = self._project_representation(
            predicted_rep, num_variables, means, stdev,
            incident_context=incident_context)
        forecast_mask = self._forecast_mask(
            predicted_rep[:, :num_variables, :])
        reconstruction = None
        if self.use_reconstruction:
            reconstruction = self.reconstruction_head(
                online_context[:, :num_variables, :]).transpose(1, 2)

        target_input = self._normalized_view(target_x)
        with torch.no_grad():
            target_rep, _ = self._encode(target_input, target_mark, target=True)
            target_rep, _, _ = self._fuse_graph(target_rep, target_input, target=True)

        predicted_variables = predicted_rep[:, :num_variables, :]
        target_variables = target_rep[:, :num_variables, :]

        jepa_loss = F.mse_loss(
            F.layer_norm(predicted_variables, predicted_variables.shape[-1:]),
            F.layer_norm(target_variables, target_variables.shape[-1:]))
        topo_loss = x_enc.new_zeros(())
        if self.topo_weight > 0:
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
        result = (forecast, jepa_loss, topo_loss, text_loss, alignment_loss)
        if return_pretrain_aux:
            return (*result, {
                'forecast_mask': forecast_mask,
                'reconstruction': reconstruction,
            })
        return (*result, forecast_mask) if return_forecast_mask else result

    def _forecast_mask(self, variable_representation):
        """Context-conditioned forecast weights/probabilities, shaped [B,S,N]."""
        if not self.use_forecast_mask:
            return None
        probabilities = torch.sigmoid(
            self.forecast_mask_head(variable_representation)).permute(0, 2, 1)
        weights = (self.forecast_mask_floor +
                   (1.0 - self.forecast_mask_floor) * probabilities)
        return {'weights': weights, 'probabilities': probabilities}

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
                                means=None, stdev=None,
                                incident_context=None):
        output = self.projector(representation).permute(0, 2, 1)
        output = output[:, :, :num_variables]
        incident_effect = self._incident_decay(
            incident_context, output.shape[1])
        if incident_effect is not None:
            output = output + incident_effect
        if self.use_norm:
            output = output * stdev[:, 0, :].unsqueeze(1)
            output = output + means[:, 0, :].unsqueeze(1)
        return output

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                 incident_data=None):
        x_enc, means, stdev = self._normalize_with_stats(x_enc)
        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out, attns = self._encode(x_enc, x_mark_enc)
        enc_out, _, _ = self._fuse_graph(enc_out, x_enc, target=False)
        enc_out, incident_context = self._fuse_incident(
            enc_out, N, incident_data)
        if self.model_variant in ('predictor', 'jepa'):
            enc_out = self.predictor(enc_out)

        # B N E -> B N S -> B S N
        dec_out = self._project_representation(
            enc_out, N, means, stdev,
            incident_context=incident_context)

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                incident_data=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec,
            incident_data=incident_data)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
