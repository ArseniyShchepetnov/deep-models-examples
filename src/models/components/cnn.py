"""Convolutional components."""
from typing import List, Optional

from torch import nn


def cnn_path(channels: List[int],
             batch_norm: bool = True,
             dropout: Optional[float] = None,
             transpose: bool = False,
             **kwargs) \
        -> nn.Module:
    """
    Generate contracting path.

    Sample is upsampled and grow number of channels by power of 2.
    """
    conv: nn.Module
    layers: List[nn.Module] = []
    for in_channels, out_channels in zip(channels[:-1], channels[1:]):
        if transpose:
            conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        else:
            conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        layers.append(conv)

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU())

        if dropout:
            layers.append(nn.Dropout2d(dropout))

    return nn.Sequential(*layers)
