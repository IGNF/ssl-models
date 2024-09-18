"""FLAIR data module."""

import datetime as dt
import json
from pathlib import Path

import albumentations as A  # noqa: N812
import numpy as np
import pandas as pd
import rasterio
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ssl_models.data.utils import (
    average_monthly_image,
    filter_masks,
    get_day_number_in_year,
)


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
        s_patch_size: int,
        filter_mask: bool,
        average_month: bool,
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
        train_dataset = FLAIRDataset(
            train_path,
            root_dir,
            ssl_phase,
            transform,
            s_patch_size,
            filter_mask,
            average_month,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_path = Path(root_dir) / "val.csv"
        val_dataset = FLAIRDataset(
            val_path,
            root_dir,
            ssl_phase,
            transform,
            s_patch_size,
            filter_mask,
            average_month,
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


class FLAIRDataset(Dataset):
    """FLAIR dataset."""

    NUM_CLASSES: dict[str, int] = {  # noqa: RUF012
        "flair": 19,
    }
    NORM_FAC_AERIAL: int = 255
    NORM_FAC_SENTINEl: int = 1
    MISSING_VAL: str = ""

    IN_CHANS = 5

    MODALITIES: dict[dict] = {  # noqa: RUF012
        "aerial": {"path": "aerial", "in_chans": 4, "norm_factor": 255.0},
    }

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        ssl_phase: str,
        transform: A.Compose | None,
        s_patch_size: int,
        filter_mask: bool,
        average_month: bool,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.s_patch_size = s_patch_size
        self.filter_mask = filter_mask
        self.average_month = average_month

        match ssl_phase:
            case "pretrain":
                csv_data = pd.read_csv(csv_path)
            case "probe" | "finetune":
                csv_data = pd.read_csv(csv_path)
                csv_data = csv_data[(csv_data["target"] != self.MISSING_VAL)]
        self.file_name = csv_data["image"].tolist()
        self.targets = csv_data["target"].tolist()
        with (
            self.root_dir / "aux-data" / "aux-data_flair-2_centroids_sp_to_patch.json"
        ).open() as file:
            self.s_centroids = json.load(file)

    def read_dates(self, txt_file: str) -> np.array:
        with Path(txt_file).open() as f:
            products = f.read().splitlines()
        s_dates_diff = get_day_number_in_year(products, 2)
        s_dates = [
            dt.datetime.strptime(x.split("_")[2][:8], "%Y%m%d").replace(
                tzinfo=dt.timezone.utc,
            )
            for x in products
        ]
        return np.array(s_dates_diff), np.array(s_dates)

    def read_superarea_and_crop(
        self,
        numpy_file: str,
        centroid: list,
    ) -> np.ndarray:
        data = np.load(numpy_file, mmap_mode="r")
        return data[
            :,
            :,
            centroid[0] - int(self.s_patch_size / 2) : centroid[0]
            + int(self.s_patch_size / 2),
            centroid[1] - int(self.s_patch_size / 2) : centroid[1]
            + int(self.s_patch_size / 2),
        ]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get dataset item."""
        output = {}

        target = rasterio.open(self.root_dir / self.targets[idx]).read()
        target = target.astype(np.float32)

        image = rasterio.open(self.root_dir / self.file_name[idx]).read()
        image = image.astype(np.float32) / self.NORM_FAC_AERIAL

        if self.transform is not None:
            image = image.transpose([1, 2, 0])
            target = target.transpose([1, 2, 0])
            transformed = self.transform(image=image, mask=target)
            image = transformed["image"].transpose([2, 0, 1])
            target = transformed["mask"].transpose([2, 0, 1])

        output.update({"aerial": image, "target": target})

        s_file_path = (
            self.root_dir / self.file_name[idx].replace("aerial", "sentinel")
        ).parents[0]
        s_images_path = next(s_file_path.glob("*_data.npy"))
        s_masks_path = next(s_file_path.glob("*_masks.npy"))
        s_products_path = next(s_file_path.glob("*_products.txt"))

        s_dates_diff, s_dates = self.read_dates(s_products_path)
        s_image = self.read_superarea_and_crop(
            s_images_path,
            self.s_centroids[Path(self.file_name[idx]).name],
        )
        s_mask = self.read_superarea_and_crop(
            s_masks_path,
            self.s_centroids[Path(self.file_name[idx]).name],
        )
        s_mask = s_mask.astype(int)

        if self.filter_mask:
            dates_to_keep = filter_masks(s_mask)
            s_image = s_image[dates_to_keep]
            s_dates_diff = s_dates_diff[dates_to_keep]
            s_dates = s_dates[dates_to_keep]

        if self.average_month:
            s_image, s_dates_diff = average_monthly_image(
                s_image,
                s_dates,
                s_dates_diff,
            )

        output.update({"s2": s_image, "s2_dates": s_dates_diff})
        return output

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_name)
