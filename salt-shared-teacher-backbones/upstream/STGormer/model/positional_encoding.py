import math

import torch
from torch import nn


class Positional1DEncoding(nn.Module):
    """Dependency-free sinusoidal encoding used by the STGormer adapter.

    The original repository obtains the same fixed sine/cosine encoding from
    the ``positional_encodings`` package. Keeping the calculation local avoids
    an otherwise unnecessary runtime dependency on the server.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        # input_data: [B, T, D]
        _, num_times, num_features = input_data.shape
        positions = torch.arange(
            num_times, dtype=input_data.dtype, device=input_data.device
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0, num_features, 2,
                dtype=input_data.dtype,
                device=input_data.device,
            ) * (-math.log(10000.0) / num_features)
        )
        encoding = torch.zeros(
            (1, num_times, num_features),
            dtype=input_data.dtype,
            device=input_data.device,
        )
        encoding[..., 0::2] = torch.sin(positions * frequencies)
        odd_width = encoding[..., 1::2].shape[-1]
        encoding[..., 1::2] = torch.cos(positions * frequencies[:odd_width])
        return input_data + encoding, encoding
