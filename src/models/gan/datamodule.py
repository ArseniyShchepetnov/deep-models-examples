"""Data module."""
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from pytorch_lightning import LightningDataModule


# pylint: disable=too-many-instance-attributes
class MNISTDataModule(LightningDataModule):
    """Data module for MNIST dataset."""

    def __init__(self,  # pylint: disable=too-many-arguments
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
                    transforms.Normalize((0.5,), (0.5, ))
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
            dataset = MNIST(root=self.data_path,
                            train=True,
                            download=True,
                            transform=self.transform)
            val_len = int(len(dataset) * self.train_val_split)
            length = [len(dataset) - val_len, val_len]
            self.train, self.val = random_split(dataset, length)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = MNIST(root=self.data_path,
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

# pylint: disable=too-many-instance-attributes


class CIFAR10DataModule(LightningDataModule):
    """Data module for CIFAR10 dataset."""

    def __init__(self,  # pylint: disable=too-many-arguments
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
