"""Test GAN classes."""
import torch

from src.models.dcgan import Discriminator, Generator, DCGAN


def test_discriminator_28():
    """Test discriminator for 28x28 image with 1 channel"""
    discriminator = Discriminator(channels=32,
                                  img_channels=1,
                                  img_size=28,
                                  max_conv=4)
    image = torch.ones(4, 1, 28, 28)  # pylint:disable=no-member
    result = discriminator(image)
    assert result.dim() == 2
    assert result.size(0) == 4
    assert result.size(1) == 1


def test_discriminator_32():
    """Test discriminator for 32x32 image with 3 channels"""
    discriminator = Discriminator(channels=32,
                                  img_channels=3,
                                  img_size=32,
                                  max_conv=5)
    image = torch.ones(4, 3, 32, 32)  # pylint:disable=no-member
    result = discriminator(image)
    assert result.dim() == 2
    assert result.size(0) == 4
    assert result.size(1) == 1


def test_generator_28():
    """Test generator for 28x28 image with 1 channel"""
    generator = Generator(channels=512,
                          img_channels=1,
                          img_size=28,
                          max_conv=5)
    inp = torch.ones(4, 100, 1, 1) / 2  # pylint:disable=no-member
    result = generator(inp)
    assert result.dim() == 4
    assert result.size()[0] == 4
    assert result.size()[1] == 1
    assert result.size()[2] == 28
    assert result.size()[3] == 28


def test_generator_32():
    """Test generator for 32x32 image with 3 channels"""
    generator = Generator(channels=512,
                          img_channels=3,
                          img_size=32,
                          max_conv=5)
    inp = torch.ones(4, 100, 1, 1) / 2  # pylint:disable=no-member
    result = generator(inp)
    assert result.dim() == 4
    assert result.size()[0] == 4
    assert result.size()[1] == 3
    assert result.size()[2] == 32
    assert result.size()[3] == 32


def test_dcgan_28():
    """Test dcgan for 28x28 image with 1 channel"""
    dcgan = DCGAN(noise_size=10,
                  channels=512,
                  img_size=28,
                  img_channels=1)
    inp = torch.ones(8, 10, 1, 1) / 2  # pylint:disable=no-member
    result = dcgan(inp)
    print(result.size())
    assert result.dim() == 4
    assert result.size(0) == 8
    assert result.size(1) == 1
    assert result.size(2) == 28
    assert result.size(3) == 28
