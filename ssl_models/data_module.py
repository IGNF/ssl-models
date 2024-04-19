"""
LightningDataModule for SSL.
"""

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


class SSLDataModule(Input):
    def __init__(
        self,
        root_dir: str,
        train_dataset: str,
        val_dataset: str,
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

        fit_params = {
            **DEFAULT_PARAMS,
            "dataloader_options": {
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
            "input_file": train_dataset,
            "root_dir": root_dir,
            "transform": transform,
        }
        validate_params = {
            **DEFAULT_PARAMS,
            "dataloader_options": {"batch_size": batch_size},
            "input_file": val_dataset,
            "root_dir": root_dir,
            "transform": transform,
        }
        super().__init__(fit_params=fit_params, validate_params=validate_params)
