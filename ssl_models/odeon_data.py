"""LightningDataModule for SSL, based on Odeon."""

from pathlib import Path

import albumentations as A  # noqa: N812
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


class OdeonData(Input):
    """GeoPile data module with Odeon."""

    NUM_CLASSES: dict[str, int] = {  # noqa: RUF012
        "mlrsnet": 46,
        "patternnet": 38,
        "rsd46-whu": 46,
        "resisc45": 45,
    }
    NORM_FAC: int = 255.0
    MISSING_VAL: int = -1

    def __init__(
        self,
        ssl_phase: str,
        root_dir: str,
        batch_size: int,
        num_workers: int,
        use_transform: bool,
    ) -> None:
        match ssl_phase:
            case "pretrain":
                train_dataset = Path(root_dir) / "train_odeon.csv"
                val_dataset = Path(root_dir) / "val_odeon.csv"
            case "probe" | "finetune":
                train_dataset = Path(root_dir) / "train_odeon_classif.csv"
                val_dataset = Path(root_dir) / "val_odeon_classif.csv"
            case _:
                raise ValueError

        if use_transform:
            transform = [
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        else:
            transform = None

        params = {
            **DEFAULT_PARAMS,
            "root_dir": root_dir,
            "transform": transform,
            "dataloader_options": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle": True,
            },
        }
        super().__init__(
            fit_params={**params, "input_file": train_dataset},
            validate_params={**params, "input_file": val_dataset},
        )
