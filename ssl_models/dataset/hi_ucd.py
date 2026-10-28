"""Hi-UCD dataset module."""

from pathlib import Path
from typing import Literal

import numpy as np

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import read_csv, strs_datetimes


class HiUCDDataset(GenericDataset):
    """Hi-UCD dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_crop: bool,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_crop=random_crop,
        )

        csv_data = read_csv(
            csv_dir=root_dir,
            stage=stage,
            ssl_phase=ssl_phase,
        )

        self.image_1 = csv_data["image_1"].to_list()
        self.image_2 = csv_data["image_2"].to_list()
        self.targets = csv_data["target"].to_list()

        self.root_dir = root_dir
        self.base_length = len(self.targets)
        self.repeats = dataset.repeats

    def build_meta(self, idx: int) -> dict:
        """Return paths and dates of each modality."""
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

        return meta

    def postprocess(
        self,
        inputs: dict[str, np.ndarray],
        meta: dict,  # noqa: ARG002
        idx: int,  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Shift labels to start at 0 and set reference date."""
        inputs["ref_date"] = strs_datetimes(["2019-07-01"])

        for target in ("hi_ucd_change_detect", "hi_ucd_seg_1", "hi_ucd_seg_2"):
            if target in inputs:
                inputs[target] = inputs[target] - 1  # to start labels at 0

        return inputs
