"""Simple GAN model."""
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(LightningDataModule):
    """Data module for CIFAR10 dataset."""

    def __init__(self,
                 data_path: str,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 train_val_split: float = 0.1,
                 transform=None):

        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        self.transform = transform

        self.train_val_split = train_val_split

        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = CIFAR10(root=self.data_path,
                              train=True,
                              download=True,
                              transform=self.transform)
            val_len = int(len(dataset) * self.train_val_split)
            length = [len(dataset) - val_len, val_len]
            self.train, self.val = random_split(dataset, length)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CIFAR10(root=self.data_path,
                                train=False,
                                download=True,
                                transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)


class Generator(nn.Module):

    def __init__(self,
                 in_size: int = 100,
                 start: int = 16,
                 img_channels: int = 3,
                 img_size: int = 32):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, start,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=True),
            nn.BatchNorm2d(start),
            nn.LeakyReLU(0.2),
        ]
        # (start, 2, 2)
        img_size = img_size // 2
        current_channels = start
        while img_size > 1:
            start = current_channels
            current_channels = start * 2

            current = [
                nn.ConvTranspose2d(start, current_channels,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.BatchNorm2d(current_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.5)
            ]
            layers += current
            img_size = img_size // 2

        layers += [
            nn.ConvTranspose2d(current_channels, img_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        res = inp
        for l in self.layers:
            res = l(res)
        return res

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):

    def __init__(self,
                 start: int = 256,
                 img_channels: int = 3,
                 img_size: int = 32):
        super().__init__()

        layers = [
            nn.Conv2d(img_channels, start,
                      kernel_size=4,
                      stride=1,
                      padding=2,
                      bias=False),
            nn.BatchNorm2d(start),
            nn.LeakyReLU(0.2),
        ]
        # (start, img_size / 2, img_size / 2)
        img_size = img_size // 2
        current_channels = start
        while img_size > 2:
            current_channels = start // 2
            current = [
                nn.Conv2d(start, current_channels,
                          kernel_size=4,
                          stride=2,
                          padding=2,
                          bias=False),
                nn.BatchNorm2d(current_channels),
                nn.LeakyReLU(0.2)
            ]
            layers += current
            img_size = img_size // 2
            start = current_channels

        layers += [
            nn.Conv2d(current_channels, 1,
                      kernel_size=4,
                      padding=0,
                      bias=False),
            nn.Sigmoid()
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        res = inp
        for l in self.layers:
            # print(l, res.size())
            res = l(res)
        res = res[:, :, 0, 0]
        return res


class GAN(LightningModule):
    def __init__(self,
                 channels):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.validation_z = torch.randn(8, 100, 1, 1)

        # self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

        self.generated_imgs = None
        self.criterion = nn.BCELoss()

    def forward(self, z: torch.Tensor):  # pylint: disable=arguments-differ
        return self.generator(z)

    def adversarial_loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        return self.criterion(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx: int):  # pylint: disable=arguments-differ
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], 100, 1, 1)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 1:

            # generate images
            synthetic = self(z)
            valid = torch.ones(imgs.size(0), 1) + torch.rand(1) * 0.2
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            score = self.discriminator(synthetic)
            g_loss = self.adversarial_loss(score, valid)
            self.log("g_loss", g_loss, prog_bar=True)

            sample_imgs = synthetic[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image(
                "generated_images", grid, batch_idx)
            return g_loss

        # train discriminator
        if optimizer_idx == 0:

            valid = torch.ones(imgs.size(0), 1) - torch.rand(1) * 0.1
            valid = valid.type_as(imgs)
            score = self.discriminator(imgs)
            real_loss = self.adversarial_loss(score, valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1) + torch.rand(1) * 0.1
            fake = fake.type_as(imgs)

            synthetic = self(z).detach()
            score = self.discriminator(synthetic)
            fake_loss = self.adversarial_loss(score, fake)

            # discriminator loss is the average of these
            d_loss = real_loss + fake_loss

            self.log("d_loss", d_loss, prog_bar=True)
            self.log("real_loss", real_loss, prog_bar=False)
            self.log("fake_loss", fake_loss, prog_bar=False)
            return d_loss

    def configure_optimizers(self):
        lr = 0.0001

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_d, opt_g], []

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(
            "generated_images_val", grid, self.current_epoch)
