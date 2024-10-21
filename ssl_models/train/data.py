"""SSL data module."""

from datetime import datetime
from pathlib import Path

import albumentations as A  # noqa: N812
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from conf.datasets import DatasetsConfig

NDIM_MULTI_DATES = 5


def collate_fn(batch: list[dict]) -> dict[str, datetime | torch.Tensor]:
    """Collate function for DataLoader."""
    elem = batch[0]
    names_arrs = [name for name in elem if isinstance(elem[name], np.ndarray)]
    names_dates = [name for name in elem if name not in names_arrs]

    collated_batch = {}
    for name in names_dates:
        collated_batch[name] = [elem[name] for elem in batch]
    for name in names_arrs:
        collated_batch[name] = torch.stack(
            [torch.from_numpy(elem[name]) for elem in batch],
        )
        # this is temporarily used to remain mono-date
        if collated_batch[name].ndim == NDIM_MULTI_DATES:
            collated_batch[name] = collated_batch[name][:, 0]
    return collated_batch


class SSLDataModule(LightningDataModule):
    """SSL data module."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        ssl_phase: str,
        batch_size: int,
        num_workers: int,
        use_transform: bool,
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
            root_dir=root_dir,
            csv_path=root_dir / "train.csv",
            ssl_phase=ssl_phase,
            transform=transform,
            dataset=datasets.dataset,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_dataset = datasets.dataset_class(
            root_dir=root_dir,
            csv_path=root_dir / "val.csv",
            ssl_phase=ssl_phase,
            transform=transform,
            dataset=datasets.dataset,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        """Return train data loader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return val data loader."""
        return self.val_loader
