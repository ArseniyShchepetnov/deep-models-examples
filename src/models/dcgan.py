"""Simple GAN model."""
from typing import List, Optional, Literal, Tuple

import torch
import torchvision
from torch import nn
from pytorch_lightning import LightningModule

SoftLabelsType = Literal["fixed", "random"]


class Generator(nn.Module):
    """Generator network."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 in_size: int = 100,
                 channels: int = 512,
                 img_channels: int = 3,
                 img_size: int = 32,
                 max_conv: int = 5):

        super().__init__()

        self.img_size = img_size
        convs: List[nn.Module] = [
            nn.ConvTranspose2d(in_size, channels,  # image size: img_size
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.ReLU()
        ]

        img_size = int(img_size / 2)
        out_channels = channels
        for _ in range(max_conv - 2):
            out_channels = int(channels / 2)
            convs += [
                # image size: img_size / 2
                nn.ConvTranspose2d(channels, out_channels,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            if img_size == 1:
                break
            img_size = int(img_size / 2)
            channels = out_channels

        convs += [
            nn.ConvTranspose2d(out_channels, img_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        ]
        self.layers = nn.Sequential(*convs)
        self.last_channels = out_channels

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Reimplement forward."""
        res = self.layers(inp)
        return res[:, :, :self.img_size, :self.img_size]

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 channels: int = 32,
                 img_channels: int = 3,
                 img_size: int = 32,
                 max_conv: int = 4,
                 leaky_relu_slope: float = 0.2):
        super().__init__()

        convs: List[nn.Module] = [
            nn.Conv2d(img_channels, channels,  # image size: img_size
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(leaky_relu_slope)
        ]

        img_size = int(img_size / 2)
        out_channels = channels
        for _ in range(max_conv - 2):
            out_channels = channels * 2
            convs += [
                nn.Conv2d(channels, out_channels,  # image size: img_size / 2
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(leaky_relu_slope)
            ]
            if img_size == 1:
                break
            img_size = int(img_size / 2)
            channels = out_channels

        self.layers = nn.Sequential(
            *convs,
            nn.Conv2d(out_channels, 1,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False)
        )
        self.out_activation = nn.Sigmoid()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Reimplement forward."""
        res = self.layers(inp)
        res = res[:, :, 0, 0]
        return self.out_activation(res)


class DCGAN(LightningModule):  # pylint: disable=too-many-instance-attributes,too-many-ancestors
    """Deep Convolutional GAN"""

    def __init__(self,  # pylint: disable=too-many-arguments
                 noise_size: int = 100,
                 channels: int = 512,
                 img_size: int = 32,
                 img_channels: int = 3,
                 learning_rate: float = 0.0001,
                 adam_betas: Tuple[float, float] = (0.5, 0.999),
                 soft_labels_value: float = 0.3,
                 soft_labels: Optional[SoftLabelsType] = None):
        super().__init__()
        self.save_hyperparameters()

        self.soft_labels_value = soft_labels_value
        self.soft_labels = soft_labels
        self.noise_size = noise_size
        self.adam_betas = adam_betas
        self.learning_rate = learning_rate

        # networks
        self.generator = Generator(in_size=noise_size,
                                   channels=channels,
                                   img_size=img_size,
                                   img_channels=img_channels)
        generator_last_channels = self.generator.last_channels
        self.discriminator = Discriminator(channels=generator_last_channels,
                                           img_size=img_size,
                                           img_channels=img_channels)

        self.validation_noise = torch.randn(8,  # pylint: disable=no-member
                                            noise_size, 1, 1)

        # self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

        self.generated_imgs = None
        self.criterion = nn.BCELoss()

    def forward(self, z: torch.Tensor):  # pylint: disable=arguments-differ
        return self.generator(z)

    def adversarial_loss(self, predicted: torch.Tensor, true: torch.Tensor):
        """Calculate loss."""
        return self.criterion(predicted, true)

    def generate_true_labels(self,
                             images: torch.Tensor,
                             soft_labels: Optional[SoftLabelsType] = None) \
            -> torch.Tensor:
        """Returns tensor with true labels"""
        # pylint: disable=no-member
        labels = torch.ones(images.size(0), 1,  # type:ignore
                            dtype=images.dtype,
                            device=self.device)
        if soft_labels:
            if soft_labels == "fixed":
                labels -= self.soft_labels_value
            elif soft_labels == "random":
                random = torch.rand(1,  # type:ignore
                                    dtype=images.dtype,
                                    device=self.device)
                labels -= random * self.soft_labels_value
        return labels

    def generate_false_labels(self,
                              images: torch.Tensor,
                              soft_labels: Optional[SoftLabelsType] = None) \
            -> torch.Tensor:
        """Returns false labels."""
        # pylint: disable=no-member
        labels = torch.zeros(images.size(0), 1,  # type:ignore
                             dtype=images.dtype,
                             device=self.device)
        if soft_labels:
            if soft_labels == "fixed":
                labels += self.soft_labels_value
            elif soft_labels == "random":
                random = torch.rand(1,  # type:ignore
                                    dtype=images.dtype,
                                    device=self.device)
                labels += random * self.soft_labels_value
        return labels

    def generate_input_noise(self, images: torch.Tensor) -> torch.Tensor:
        """Generate input noise for generator."""
        # pylint: disable=no-member
        return torch.randn(images.size(0),  # type:ignore
                           self.noise_size, 1, 1,
                           dtype=images.dtype,
                           device=self.device)

    def training_step(self,  # pylint: disable=arguments-differ, too-many-locals
                      batch,
                      batch_idx,
                      optimizer_idx: int) -> torch.Tensor:
        imgs, _ = batch

        # sample noise
        input_noise = self.generate_input_noise(imgs)

        # train generator
        if optimizer_idx == 1:

            # generate images
            synthetic = self(input_noise)
            true_labels = self.generate_true_labels(imgs, soft_labels=None)

            # adversarial loss is binary cross-entropy
            score = self.discriminator(synthetic)
            g_loss = self.adversarial_loss(score, true_labels)
            self.log("generator_loss", g_loss, prog_bar=True)

            sample_imgs = synthetic[:6]
            grid = torchvision.utils.make_grid(sample_imgs)

            self.logger.experiment.add_image(  # type: ignore
                "generated_images", grid, batch_idx)
            loss = g_loss

        # train discriminator
        if optimizer_idx == 0:

            true_labels = self.generate_true_labels(imgs, self.soft_labels)
            score = self.discriminator(imgs)
            real_loss = self.adversarial_loss(score, true_labels)

            # how well can it label as fake?
            false_labels = self.generate_false_labels(imgs, self.soft_labels)
            synthetic = self(input_noise).detach()
            score = self.discriminator(synthetic)
            fake_loss = self.adversarial_loss(score, false_labels)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            self.log("discriminator_loss", d_loss, prog_bar=True)
            self.log("real_loss", real_loss, prog_bar=False)
            self.log("fake_loss", fake_loss, prog_bar=False)
            loss = d_loss

        return loss

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.discriminator.parameters(),
                             lr=self.learning_rate,
                             betas=self.adam_betas),
            torch.optim.Adam(self.generator.parameters(),
                             lr=self.learning_rate,
                             betas=self.adam_betas)
        ]
        return optimizers, []

    def on_validation_epoch_end(self):
        noise = self.validation_noise.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(noise)
        grid = torchvision.utils.make_grid(sample_imgs)
        # type: ignore
        self.logger.experiment.add_image("generated_images_val",
                                         grid, self.current_epoch)
