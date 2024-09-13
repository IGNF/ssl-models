"""TreeSatAI-Time-Series data module."""

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class TreeSatAITSData(LightningDataModule):
    """TreeSatAI-Time-Series data module."""
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

        train_path = Path(root_dir) / "train.csv"
        train_dataset = TreeSatAITSDataset(
            train_path,
            root_dir,
            ssl_phase,
            transform,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_path = Path(root_dir) / "val.csv"
        val_dataset = TreeSatAITSDataset(
            val_path,
            root_dir,
            ssl_phase,
            transform,
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


class TreeSatAITSDataset(Dataset):
    """TreeSatAI-Time-Series dataset."""
    NORM_FAC: int = 255.0
    NUM_CLASSES: dict[str, int] = {  # noqa: RUF012
        "treesat": 15,
    }
    MISSING_VAL: int = 0

    CLASSES = [
        "Pinus",
        "Acer",
        "Fraxinus",
        "Picea",
        "Fagus",
        "Alnus",
        "Quercus",
        "Prunus",
        "Populus",
        "Larix",
        "Cleared",
        "Pseudotsuga",
        "Tilia",
        "Betula",
        "Abies",
    ]

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
                    (csv_data[self.CLASSES] != self.MISSING_VAL).any(axis=1)
                ]

        self.file_name = csv_data["image"].tolist()
        self.targets = {
            dataset: csv_data[self.CLASSES].values.tolist()
            for dataset in self.NUM_CLASSES
        }

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get dataset item."""
        file_path = self.root_dir / self.file_name[idx]
        image = rasterio.open(file_path).read()
        image = image.astype(np.float32) / self.NORM_FAC
        if self.transform is not None:
            image = image.transpose([1, 2, 0])
            image = self.transform(image=image)["image"]
            image = image.transpose([2, 0, 1])

        targets = {
            f"target_{dataset}": target[idx]
            for dataset, target in self.targets.items()
        }

        return {"image": image[:3], **targets}  # TODO 4 channels au lieu de 3

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_name)

