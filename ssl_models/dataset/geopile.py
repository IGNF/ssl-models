"""GeoPile data module."""

from pathlib import Path
from typing import Literal

import albumentations as A  # noqa: N812
import numpy as np
import pandas as pd
from torch import Tensor

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset


class GeoPileDataset(GenericDataset):
    """GeoPile dataset."""

    MISSING_VAL = -1

    def __init__(
        self,
        root_dir: Path,
        csv_path: Path,
        transform: A.Compose | None,
        dataset: DatasetConfig,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> None:
        super().__init__(transform=transform, dataset=dataset)

        name_targets = dataset.filter_targets
        col_targets = [
            f"class_index_{name_target.replace('_', '-')}"
            for name_target in name_targets
        ]
        match ssl_phase:
            case "pretrain":
                csv_data = pd.read_csv(csv_path)
            case "probe" | "finetune":
                csv_data = pd.read_csv(csv_path)
                csv_data = csv_data[
                    (csv_data[col_targets] != self.MISSING_VAL).any(axis=1)
                ]

        self.root_dir = Path(root_dir)
        self.file_name = csv_data["image"].tolist()
        self.targets = {
            name_target: csv_data[col_target].tolist()
            for name_target, col_target in zip(name_targets, col_targets)
        }

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get dataset item."""
        inputs = {}
        for name_target, target in self.targets.items():
            inputs[name_target] = np.array(target[idx])
            inputs[f"{name_target}_dates"] = [None]

        meta = {}
        meta["rgb_path"] = self.root_dir / self.file_name[idx]
        meta["rgb_dates"] = [None]

        inputs_raster = super().preprocess_rasters(meta)
        inputs.update(inputs_raster)
        return self.transform(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_name)
