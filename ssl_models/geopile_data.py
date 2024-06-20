"""LightningDataModule for SSL."""

from pathlib import Path

import albumentations as A  # noqa: N812
import numpy as np
import pandas as pd
import rasterio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class GeoPileData(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        use_tranform: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
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

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class GeoPileDataset(Dataset):
    num_classes = {
        "mlrsnet": 46,
        "patternnet": 38,
        "rsd46-whu": 46,
        "resisc45": 45,
    }
    norm_fac = 255.0
    missing_val = -1

    def __init__(self, csv_path, root_dir, transform):
        self.root_dir = Path(root_dir)
        self.transform = transform

        csv_data = pd.read_csv(csv_path)
        self.file_name = csv_data["image"].tolist()
        self.targets = {
            dataset: csv_data[f"class_index_{dataset}"].tolist()
            for dataset in self.num_classes
        }

    def __getitem__(self, idx):
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

    def __len__(self):
        return len(self.file_name)
