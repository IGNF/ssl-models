"""GeoPile data module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import albumentations as A  # noqa: N812
import numpy as np
import pandas as pd
import rasterio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from torch import Tensor


class GeoPileData(LightningDataModule):
    """GeoPile data module."""

    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        num_workers: int,
        *,
        use_tranform: bool = True,
    ) -> None:
        super().__init__()
        if use_tranform:
            transform = A.Compose(
                [
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
            )
        else:
            transform = None

        train_path = Path(root_dir) / "train.csv"
        train_dataset = GeoPileDataset(train_path, root_dir, transform)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        val_path = Path(root_dir) / "val.csv"
        val_dataset = GeoPileDataset(val_path, root_dir, transform)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Return train data loader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return val data loader."""
        return self.val_loader


class GeoPileDataset(Dataset):
    """GeoPile dataset."""

    num_classes: ClassVar[dict[str, int]] = {
        "mlrsnet": 46,
        "patternnet": 38,
        "rsd46-whu": 46,
        "resisc45": 45,
    }
    norm_fac: ClassVar[int] = 255.0
    missing_val: ClassVar[int] = -1

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        transform: A.Compose | None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        csv_data = pd.read_csv(csv_path)
        self.file_name = csv_data["image"].tolist()
        self.targets = {
            dataset: csv_data[f"class_index_{dataset}"].tolist()
            for dataset in self.num_classes
        }

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get dataset item."""
        file_path = self.root_dir / self.file_name[idx]
        image = rasterio.open(file_path).read()
        image = image.astype(np.float32) / self.norm_fac
        if self.transform is not None:
            image = image.transpose([1, 2, 0])
            image = self.transform(image=image)["image"]
            image = image.transpose([2, 0, 1])

        targets = {
            f"target_{dataset}": target[idx] for dataset, target in self.targets.items()
        }
        return {"image": image, **targets}

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_name)
