"""FLAIR data module."""

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

class FLAIRData(LightningDataModule):
    """FLAIR data module."""

    IMG_SIZE: int = 256

    def __init__(
        self,
        ssl_phase: str,
        root_dir: str,
        batch_size: int,
        num_workers: int,
        use_transform: bool,
    ) -> None:
        super().__init__()
        if use_transform:
            transform = A.Compose(
                [
                    A.Resize(self.IMG_SIZE, self.IMG_SIZE),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
            )
        else:
            transform = None

        train_path = Path(root_dir) / "train_toy.csv"
        train_dataset = FLAIRDataset(train_path, root_dir, ssl_phase, transform)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_path = Path(root_dir) / "val_toy.csv"
        val_dataset = FLAIRDataset(val_path, root_dir, ssl_phase, transform)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        """return train data loader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """return val data loader."""
        return self.val_loader



class FLAIRDataset(Dataset):
    """FLAIR dataset."""

    NUM_CLASSES: dict[str, int] = {  # noqa: RUF012
        "flair": 19,
    }
    NORM_FAC: int = 255.0
    MISSING_VAL: str = ""

    IN_CHANS = 5

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        ssl_phase: str,
        transform: A.Compose | None,
    ) -> None:

        self.root_dir = Path(root_dir)
        self.transform = transform

        match ssl_phase:
            case "pretrain":
                csv_data = pd.read_csv(csv_path)
            case "probe" | "finetune":
                csv_data = pd.read_csv(csv_path)
                csv_data = csv_data[
                    (csv_data["target"] != self.MISSING_VAL)
                ]
        self.file_name = csv_data["image"].tolist()
        self.targets = {
            dataset: csv_data["target"].values.tolist()
            for dataset in self.NUM_CLASSES
        }

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get dataset item."""
        file_path = self.root_dir / self.file_name[idx]
        image = rasterio.open(file_path).read()
        image = image.astype(np.float32) / self.NORM_FAC

        targets = {
            f"target_{dataset}": rasterio.open(
                self.root_dir / target[idx]).read()
            for dataset, target in self.targets.items()
        }

        if self.transform is not None:
            image = image.transpose([1, 2, 0])
            image = self.transform(image=image)["image"]
            image = image.transpose([2, 0, 1])

            targets = {
                dataset: self.transform(
                    image=target.transpose([1, 2, 0])
                )["image"].transpose([2, 0, 1])
                for dataset, target in targets.items()
            }

        return {"image": image, **targets}

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_name)