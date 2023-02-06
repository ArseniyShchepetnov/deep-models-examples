"""Variational autoencoder."""
from typing import List, Optional, Tuple

import torch
import torchvision
from pytorch_lightning import LightningModule
from torch import nn

from src.models.components.cnn import cnn_path


class VaeEncoder(nn.Module):
    """Encoder part."""

    def __init__(self,
                 channels: Optional[List[int]] = None,
                 img_channels: int = 1,
                 latent_size: int = 100,
                 dropout: Optional[float] = None):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64, 128, 256]

        layers = [
            nn.Conv2d(img_channels, channels[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.LeakyReLU(),
        ]

        convs = cnn_path(channels=channels,
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         bias=False,
                         dropout=dropout)
        layers.append(convs)

        self.layers = nn.Sequential(*layers)

        self.latent_mean = nn.Linear(channels[-1], latent_size)
        self.latent_log_var = nn.Linear(channels[-1], latent_size)
        self.latent_size = latent_size

    def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reimplement forward pass."""
        print(inp.size())
        res = self.layers(inp)
        print(res.size())
        flat = torch.flatten(res, start_dim=1)
        mean = self.latent_mean(flat)
        log_var = self.latent_log_var(flat)
        return mean, log_var


class VaeDecoder(nn.Module):
    """Decoder part."""

    def __init__(self,
                 channels: Optional[List[int]] = None,
                 latent_size: int = 100,
                 img_size: int = 28,
                 img_channels: int = 1,
                 dropout: Optional[float] = None):
        super().__init__()

        if channels is None:
            channels = [256, 128, 64, 32]

        self.latent_input = nn.Linear(latent_size, channels[0] * 4)
        self.layers = cnn_path(channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=False,
                               transpose=True,
                               dropout=dropout)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(channels[-1], img_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Tanh()
        )

        self.img_size = img_size
        self.latent_size = latent_size
        self.first_channel = channels[0]

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Reimplement forward pass."""
        img: torch.Tensor = self.latent_input(inp)
        img = img.view(-1, self.first_channel, 2, 2)
        img = self.layers(img)
        res = self.last(img)
        return res[:, :, :self.img_size, :self.img_size]


def kld_loss(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL divergence."""
    return (-1 - log_var + mean ** 2 + log_var.exp()).mean()


class CnnVae(LightningModule):  # pylint: disable=too-many-ancestors
    """Variational autoencoder with CNN layers."""

    def __init__(self,
                 img_size: int = 28,
                 img_channels: int = 1,
                 latent_size: int = 100,
                 learning_rate: float = 0.0005,
                 kld_weight: float = 0.00025,
                 dropout: Optional[float] = None,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = VaeEncoder(latent_size=latent_size,
                                  dropout=dropout,
                                  img_channels=img_channels)
        self.decoder = VaeDecoder(latent_size=latent_size,
                                  dropout=dropout,
                                  img_channels=img_channels)

        self.latent_size = latent_size
        self.img_loss = nn.MSELoss()
        self.learning_rate = learning_rate
        self.kld_weight = kld_weight
        self.img_size = img_size

        self.validation_noise = torch.randn(8, latent_size, 1, 1)

    # pylint: disable = arguments-differ
    def forward(self, imgs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Return reconstructed image and latent variables.
        """
        mean, log_var = self.encoder(imgs)
        noise = self.reparametrization(mean, log_var)
        return self.decoder(noise), mean, log_var

    @ staticmethod
    def reparametrization(mean: torch.Tensor,
                          logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)"""
        std = torch.exp(logvar)
        noise = torch.randn_like(std)
        return noise * std + mean

    def sample(self, noise: torch.Tensor) -> torch.Tensor:
        """Sample images from noise."""
        return self.decoder(noise)

    def loss(self,
             images: torch.Tensor,
             synthetic: torch.Tensor,
             mean: torch.Tensor,
             log_var: torch.Tensor) -> torch.Tensor:
        """VAE loss: KL + MSE."""
        kl_div = kld_loss(mean, log_var)
        img_loss = self.img_loss(synthetic, images)
        self.log("kld_loss", kl_div, prog_bar=False)
        self.log("img_loss", img_loss, prog_bar=False)
        return kl_div * self.kld_weight + img_loss

    def training_step(self,  # pylint: disable=arguments-differ
                      batch,
                      batch_idx) -> torch.Tensor:
        imgs, _ = batch
        synthetic, mean, log_var = self.forward(imgs)
        sample_imgs = synthetic[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images",  # type: ignore
                                         grid,
                                         batch_idx)
        loss = self.loss(imgs, synthetic, mean, log_var)
        self.log("loss", loss, prog_bar=True)
        self.log("mu", mean.mean(), prog_bar=False)
        self.log("exp(sigma)", log_var.mean(), prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.parameters(), lr=self.learning_rate),
        ]
        return optimizers, []

    def on_validation_epoch_end(self):
        noise = self.validation_noise.type_as(self.generator.model[0].weight)
        sample_imgs, _, _ = self(noise)
        grid = torchvision.utils.make_grid(sample_imgs)
        # type: ignore
        self.logger.experiment.add_image("validation_images",
                                         grid, self.current_epoch)
