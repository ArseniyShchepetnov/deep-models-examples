"""Test utilities for cnn construction."""
import torch

from src.models.components.cnn import cnn_path


def test_cnn_path():
    """Test contracting2d sizes expectations."""

    for img_size in [32]:
        img = torch.zeros(5, 2, img_size, img_size)
        layers = cnn_path(channels=[2, 4, 8],
                          kernel_size=4,
                          stride=2,
                          padding=1)
        result: torch.Tensor = layers(img)
        print(img.size())
        print(result.size())
        assert result.dim() == 4
        assert result.size() == torch.Size([5,
                                            8,
                                            img_size // 4,
                                            img_size // 4])
