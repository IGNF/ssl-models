"""Hi-UCD dataset module."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import strs_datetimes


class HiUCDDataset(GenericDataset):
    """Hi-UCD dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_crop: bool,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_crop=random_crop,
        )

        csv_path = root_dir / f"{stage}.csv"
        csv_data = pd.read_csv(csv_path)

        self.image_1 = csv_data["image_1"].to_list()
        self.image_2 = csv_data["image_2"].to_list()
        self.targets = csv_data["target"].to_list()

        self.root_dir = root_dir
        self.base_length = len(self.targets)
        self.repeats = dataset.repeats

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        idx, start_gcd = self.sample_gcd(
            idx,
            base_length=self.base_length,
            repeats=self.repeats,
        )

        meta = {}

        meta["image_1_path"] = self.root_dir / self.image_1[idx]
        meta["image_1_dates"] = strs_datetimes(["2018-07-01"])

        meta["image_2_path"] = self.root_dir / self.image_2[idx]
        meta["image_2_dates"] = strs_datetimes(["2019-07-01"])

        meta["hi_ucd_change_detect_path"] = self.root_dir / self.targets[idx]
        meta["hi_ucd_change_detect_dates"] = strs_datetimes(["2019-07-01"])

        meta["hi_ucd_seg_1_path"] = self.root_dir / self.targets[idx]
        meta["hi_ucd_seg_1_dates"] = strs_datetimes(["2019-07-01"])

        meta["hi_ucd_seg_2_path"] = self.root_dir / self.targets[idx]
        meta["hi_ucd_seg_2_dates"] = strs_datetimes(["2019-07-01"])

        inputs = self.preprocess_rasters(meta)
        inputs["ref_date"] = strs_datetimes(["2019-07-01"])

        for target in ["hi_ucd_change_detect", "hi_ucd_seg_1", "hi_ucd_seg_2"]:
            if target in inputs:
                inputs[target] = inputs[target] - 1  # to start labels at 0

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2
