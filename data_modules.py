import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size : int,
            num_workers: int,
            pin_memory : bool = False,
            drop_last : bool = False,
            data_dir : str = "./",
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size,
        self.num_workers = num_workers,
        self.pin_memory = pin_memory,
        self.drop_last = drop_last

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, (0.3001)))])

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=True, transform=self.transform)

    def train_dataloader(self) -> DataLoader: 
        return DataLoader(
            self.mnist_train,
            batch_size = self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size = self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size = self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True
        )