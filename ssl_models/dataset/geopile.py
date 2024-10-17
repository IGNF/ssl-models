"""GeoPile data module."""

from pathlib import Path

import albumentations as A  # noqa: N812
import numpy as np
import pandas as pd
import rasterio
from torch import Tensor
from torch.utils.data import Dataset


class GeoPileDataset(Dataset):
    """GeoPile dataset."""

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
                class_cols = [f"class_index_{dataset}" for dataset in self.NUM_CLASSES]
                csv_data = pd.read_csv(csv_path)
                csv_data = csv_data[
                    (csv_data[class_cols] != self.MISSING_VAL).any(axis=1)
                ]

        self.file_name = csv_data["image"].tolist()
        self.targets = {
            dataset: csv_data[f"class_index_{dataset}"].tolist()
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
            f"target_{dataset}": target[idx] for dataset, target in self.targets.items()
        }
        return {"image": image, **targets}

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_name)
