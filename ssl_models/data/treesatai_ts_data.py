"""TreeSatAI-Time-Series data module."""

import datetime as dt
from pathlib import Path

import albumentations as A  # noqa: N812
import h5py
import numpy as np
import pandas as pd
import rasterio
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ssl_models.data.utils import (
    average_monthly_image,
    clip_ts,
    filter_masks,
    get_day_number_in_year,
)


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
        modality: str,
        filter_mask: bool,
        average_month: bool,
        clip_ts_val: int,
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
            modality,
            filter_mask,
            average_month,
            clip_ts_val,
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
            modality,
            filter_mask,
            average_month,
            clip_ts_val,
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
    IN_CHANS = 4

    CLASSES: list[str] = [  # noqa: RUF012
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

    MODALITIES: dict[dict] = {  # noqa: RUF012
        "aerial": {"path": "aerial", "in_chans": 4, "norm_factor": 255.0},
        "s2-60m-mono": {
            "path": "sentinel/s2/200m",
            "in_chans": 10,
            "norm_factor": 1,
        },
        "s2-200m-mono": {
            "path": "sentinel/s2/60m",
            "in_chans": 10,
            "norm_factor": 1,
        },
        "s1-60m-mono": {
            "path": "sentinel/s1/60m",
            "in_chans": 2,
            "norm_factor": 1,
        },
        "s1-200m-mono": {
            "path": "sentinel/s1/200m",
            "in_chans": 2,
            "norm_factor": 1,
        },
        "s2-ts": {
            "path": "sentinel-ts",
            "in_chans": None,
            "norm_factor": 1,
        },
        "s1-ts": {
            "path": "sentinel-ts",
            "in_chans": 2,
            "norm_factor": 1,
        },
    }

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        ssl_phase: str,
        transform: A.Compose | None,
        modality: str,
        filter_mask: bool,
        average_month: bool,
        clip_ts_val: int,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.modality = modality
        self.filter_mask = filter_mask
        self.average_month = average_month
        self.clip_ts_val = clip_ts_val

        match ssl_phase:
            case "pretrain":
                csv_data = pd.read_csv(csv_path)
            case "probe" | "finetune":
                csv_data = pd.read_csv(csv_path)
                csv_data = csv_data[
                    (csv_data[self.CLASSES] != self.MISSING_VAL).any(axis=1)
                ]

        self.file_name = csv_data["image"].tolist()
        self.targets = csv_data[self.CLASSES].to_numpy()

    def _get_tiff_data(self, path: str, modality: str) -> Tensor:
        image = rasterio.open(path).read()
        image = image.astype(np.float32) / self.MODALITIES[modality]["norm_factor"]
        if self.transform is not None:
            image = image.transpose([1, 2, 0])
            image = self.transform(image=image)["image"]
            image = image.transpose([2, 0, 1])
        return image[: self.MODALITIES[modality]["in_chans"], :, :]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get dataset item."""
        output = {"target": self.targets[idx]}

        match self.modality:
            case (
                "aerial"
                | "s2-60m-mono"
                | "s2-200m-mono"
                | "s1-60m-mono"
                | "s1-200m-mono"
            ):
                file_path = (
                    self.root_dir
                    / Path(self.MODALITIES[self.modality]["path"])
                    / Path(self.file_name[idx])
                )
                output["image"] = self._get_tiff_data(file_path, self.modality)

            case "s1-ts":
                file_path = next(
                    iter(
                        (self.root_dir / self.MODALITIES[self.modality]["path"]).glob(
                            self.file_name[idx].split(".tif")[0] + "*.h5",
                        ),
                    ),
                )

                with h5py.File(file_path, "r") as file:
                    s1_asc_data = file["sen-1-asc-data"][:]
                    s1_asc_products = file["sen-1-asc-products"][:]
                    s1_des_data = file["sen-1-des-data"][:]
                    s1_des_products = file["sen-1-des-products"][:]

                s1_asc_dates_diff = get_day_number_in_year(s1_asc_products, 4)
                s1_des_dates_diff = get_day_number_in_year(s1_des_products, 4)
                s1_asc_dates = [
                    dt.datetime.strptime(
                        x.decode().split("_")[4][:8],
                        "%Y%m%d",
                    ).replace(tzinfo=dt.timezone.utc)
                    for x in s1_asc_products
                ]

                s1_des_dates = [
                    dt.datetime.strptime(
                        x.decode().split("_")[4][:8],
                        "%Y%m%d",
                    ).replace(tzinfo=dt.timezone.utc)
                    for x in s1_des_products
                ]

                if self.average_month:
                    s1_asc_data, s1_asc_dates_diff = average_monthly_image(
                        s1_asc_data,
                        s1_asc_dates,
                        s1_asc_dates_diff,
                    )
                    s1_des_data, s1_des_dates_diff = average_monthly_image(
                        s1_des_data,
                        s1_des_dates,
                        s1_des_dates_diff,
                    )

                if len(s1_asc_dates_diff) > self.clip_ts_val:
                    s1_asc_data, s1_asc_dates_diff = clip_ts(
                        s1_asc_data,
                        s1_asc_dates_diff,
                        self.clip_ts_val,
                    )

                if len(s1_des_dates_diff) > self.clip_ts_val:
                    s1_des_data, s1_des_dates_diff = clip_ts(
                        s1_des_data,
                        s1_des_dates_diff,
                        self.clip_ts_val,
                    )

                output.update(
                    {
                        "s1-asc-data": s1_asc_data[
                            :,
                            : self.MODALITIES[self.modality]["in_chans"],
                            :,
                            :,
                        ],
                        "s1-asc-dates": s1_asc_dates_diff,
                        "s1-des-data": s1_des_data[
                            :,
                            : self.MODALITIES[self.modality]["in_chans"],
                            :,
                            :,
                        ],
                        "s1-des-dates": s1_des_dates_diff,
                    },
                )

            case "s2-ts":
                file_path = next(
                    iter(
                        (self.root_dir / self.MODALITIES[self.modality]["path"]).glob(
                            self.file_name[idx].split(".tif")[0] + "*.h5",
                        ),
                    ),
                )
                with h5py.File(file_path, "r") as file:
                    s2_data = file["sen-2-data"][:]
                    s2_products = file["sen-2-products"][:]
                    s2_masks = file["sen-2-masks"][:]

                if self.filter_mask:
                    dates_to_keep = filter_masks(s2_masks)
                    s2_products = s2_products[dates_to_keep]
                    s2_data = s2_data[dates_to_keep]

                s2_dates_diff = get_day_number_in_year(s2_products)
                s2_dates = [
                    dt.datetime.strptime(
                        x.decode().split("_")[2][:8],
                        "%Y%m%d",
                    ).replace(tzinfo=dt.timezone.utc)
                    for x in s2_products
                ]

                if self.average_month:
                    s2_data, s2_dates_diff = average_monthly_image(
                        s2_data,
                        s2_dates,
                        s2_dates_diff,
                    )

                if len(s2_dates_diff) > self.clip_ts_val:
                    s2_data, s2_dates_diff = clip_ts(
                        s2_data,
                        s2_dates_diff,
                        self.clip_ts_val,
                    )

                output.update({"s2-data": s2_data, "s2-dates": s2_dates_diff})

        return output

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_name)
