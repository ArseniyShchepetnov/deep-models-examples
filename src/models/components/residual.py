"""Residual layers for CNN."""
from typing import List, Optional, Tuple

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Residual block implementation."""

    def __init__(self,
                 channels: int,
                 n_layers: int,
                 kernel_size: int = 3,
                 dropout: Optional[float] = None,
                 batch_norm: bool = False):

        super().__init__()

        padding = int(kernel_size / 2)

        layers: List[nn.Module] = []
        for _ in range(0, n_layers):

            conv = nn.Conv2d(in_channels=channels,
                             out_channels=channels,
                             kernel_size=kernel_size,
                             padding=padding)
            layers.append(conv)
            if dropout:
                dropout_layer = nn.Dropout2d(dropout)
                layers.append(dropout_layer)
            if batch_norm:
                layers.append(nn.BatchNorm2d(channels))

        self.layers = nn.Sequential(*layers)
        self.activation = nn.LeakyReLU(0.02)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self.layers(inp)
        add = torch.add(output, inp)
        result = self.activation(add)
        return result
