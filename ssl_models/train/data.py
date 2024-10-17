"""SSL data module."""

from pathlib import Path

import albumentations as A  # noqa: N812
import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from conf.datasets import DatasetsConfig


def batch_collate(batch):
    """Collate function for DataLoader."""
    batch_collated = {}
    for name in batch:
        if isinstance(batch[name][0], Tensor):
            batch_collated[name] = torch.stack(batch[name], dim=0)
    return batch_collated


class SSLDataModule(LightningDataModule):
    """SSL data module."""

    def __init__(
        self,
        ssl_phase: str,
        batch_size: int,
        num_workers: int,
        use_transform: bool,
        root_dir: str,
        datasets: DatasetsConfig,
    ) -> None:
        super().__init__()
        if use_transform:
            transform = A.Compose(
                [
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
            )
        else:
            transform = None

        root_dir = Path(datasets.root_dir) / datasets.dataset.rel_dir
        train_dataset = datasets.dataset_class(
            root_dir,
            root_dir / "train.csv",
            ssl_phase,
            transform,
            datasets.dataset,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_dataset = datasets.dataset_class(
            root_dir,
            root_dir / "val.csv",
            ssl_phase,
            transform,
            datasets.dataset,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Return train data loader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return val data loader."""
        return self.val_loader
