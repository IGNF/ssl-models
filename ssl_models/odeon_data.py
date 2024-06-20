"""LightningDataModule for SSL, based on Odeon."""

from pathlib import Path

import albumentations as A
from odeon.data import Input

DEFAULT_PARAMS = {
    "input_fields": {
        "image": {
            "name": "image",
            "type": "raster",
            "band_indices": (1, 2, 3),
            "dtype": "uint8",
            "mean": 0,
            "std": 1,
        },
    },
}


class SSLOdeonData(Input):
    def __init__(
        self,
        root_dir: str,
        shuffle_val: bool = True,
        use_tranform: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        if use_tranform:
            transform = [
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        else:
            transform = None

        train_dataset = Path(root_dir) / "train.csv"
        val_dataset = Path(root_dir) / "val.csv"

        params = {
            **DEFAULT_PARAMS,
            "root_dir": root_dir,
            "transform": transform,
        }
        fit_params = {
            **params,
            "dataloader_options": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle": True,
                "drop_last": True,
            },
            "input_file": train_dataset,
        }
        validate_params = {
            **params,
            "dataloader_options": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle": shuffle_val,
                "drop_last": True,
            },
            "input_file": val_dataset,
        }
        super().__init__(
            fit_params=fit_params,
            validate_params=validate_params,
        )
