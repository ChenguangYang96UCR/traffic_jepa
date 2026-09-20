"""Unmerged LoRA adapters; checkpoints include both base and adapter weights."""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('LoRA requires an unwrapped nn.Linear')
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('LoRA requires rank>0, finite alpha>0, 0<=dropout<1')
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Save scaling too, so reloading cannot silently change the learned map.
        self.register_buffer('scaling', base.weight.new_tensor(alpha / rank))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + ((self.dropout(x) @ self.lora_A.T) @ self.lora_B.T) * self.scaling
