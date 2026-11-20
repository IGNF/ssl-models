"""Guyane dataset module."""

from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import dict_datetimes, read_csv, strs_datetimes


class GuyaneDataset(GenericDataset):
    """Guyane dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_dates: bool,
        random_crop: bool,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )

        csv_data = read_csv(
            csv_dir=root_dir,
            stage=stage,
            ssl_phase=ssl_phase,
            val_pretrain=dataset.val_pretrain,
            test_pretrain=dataset.test_pretrain,
        )

        self.root_dir = root_dir
        self.patch_ids = csv_data["patch_id"].tolist()
        self.base_length = len(self.patch_ids)
        self.repeats = dataset.repeats

        dates_path = root_dir / "GLOBAL_ALL_MTD" / "GLOBAL_ALOS2_MTD_DATES.gpkg"
        dates_mod = gpd.read_file(dates_path, engine="pyogrio", use_arrow=True)
        self.dates_alos2 = {
            row["patch_id"]: row["date"] for _, row in dates_mod.iterrows()
        }

        dates_path = root_dir / "GLOBAL_ALL_MTD" / "GLOBAL_S2_MTD_DATES.gpkg"
        dates_mod = gpd.read_file(dates_path, engine="pyogrio", use_arrow=True)
        self.dates_s2 = {
            row["patch_id"]: row["acquisition_dates"] for _, row in dates_mod.iterrows()
        }

    def build_meta(self, idx: int) -> dict:
        """Return paths and dates of each modality."""
        patch_id = self.patch_ids[idx]
        domain, area, pos = patch_id.split("_")

        meta = {}
        meta["alos2_path"] = (
            self.root_dir
            / f"{domain}_ALOS2"
            / area
            / f"{domain}_ALOS2_{area}_{pos}.tif"
        )
        meta["s2_path"] = (
            self.root_dir / f"{domain}_S2" / area / f"{domain}_S2_{area}_{pos}.tif"
        )
        meta["alos2_dates"] = strs_datetimes(
            [self.dates_alos2[patch_id]],
            fmt="%Y%m%d",
        )
        meta["s2_dates"] = dict_datetimes(
            self.dates_s2[patch_id],
            start=1,
        )
        return meta

    def postprocess(
        self,
        inputs: dict[str, np.ndarray],
        meta: dict,
        idx: int,  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Add reference date."""
        inputs["ref_date"] = meta["alos2_dates"]

        return inputs
