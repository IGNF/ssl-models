"""GeoPile data module."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import strs_datetimes


class GeoPileDataset(GenericDataset):
    """GeoPile dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_dates: bool,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
        )

        csv_path = root_dir / f"{stage}.csv"
        name_targets = dataset.filter_targets
        col_targets = [
            f"class_index_{name_target.replace('_', '-')}"
            for name_target in name_targets
        ]
        missing_vals = [
            target.missing_val for name_target, target in dataset.targets.items()
        ]
        match ssl_phase:
            case "pretrain":
                csv_data = pd.read_csv(csv_path)
            case "probe" | "finetune":
                csv_data = pd.read_csv(csv_path)
                csv_data = csv_data[
                    np.any(
                        [
                            (csv_data[col] != missing_val)
                            for col, missing_val in zip(
                                col_targets,
                                missing_vals,
                            )
                        ],
                        axis=0,
                    )
                ]

        self.root_dir = Path(root_dir)
        self.file_name = csv_data["image"].tolist()
        self.targets = {
            name_target: csv_data[col_target].tolist()
            for name_target, col_target in zip(name_targets, col_targets)
        }

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        meta = {}
        meta["rgb_path"] = self.root_dir / self.file_name[idx]
        meta["rgb_dates"] = strs_datetimes(["2019-07-01"])

        inputs = self.preprocess_rasters(meta)

        for name_target, target in self.targets.items():
            inputs[name_target] = np.array(target[idx])
            inputs[f"{name_target}_dates"] = strs_datetimes(["2019-07-01"])

        inputs["ref_date"] = strs_datetimes(["2019-07-01"])

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_name)
