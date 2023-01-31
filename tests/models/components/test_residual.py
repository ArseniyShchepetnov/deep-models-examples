"""Test residual block."""
import numpy as np
import torch

from src.models.components.residual import ResidualBlock


def test_residual_default():
    """Test residual block default."""
    block = ResidualBlock(1, 2, 7)
    data = np.array([[[0, 1, 2], [1, 2, 3], [4, 3, 2]]])
    tensor = torch.Tensor(data)
    result = block(tensor)
    assert result.size() == torch.Size([2, 3, 3])
