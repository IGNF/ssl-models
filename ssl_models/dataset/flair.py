"""FLAIR dataset module."""

import json
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np

from conf.dataset.flair_inc import FLAIRIncConfig
from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import dict_datetimes, read_csv, strs_datetimes


class FLAIRDataset(GenericDataset):
    """FLAIR dataset."""

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

        csv_data = read_csv(
            root_dir=root_dir,
            stage=stage,
            ssl_phase=ssl_phase,
            balance_pretrain=dataset.balance_pretrain,
        )

        self.root_dir = root_dir.parent / FLAIRIncConfig.rel_dir
        self.patch_ids = csv_data["image"].tolist()

        self.mod_mapping = {
            "aerial": "aerial_rgbi",
            "s2": "sentinel2_ts",
            "s2_mask": "sentinel2_msk-sc",
            "flair_cosia": "aerial_label_cosia",
        }

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        domain, area, pos = self.patch_ids[idx].split("_")

        meta = {}
        for name_mod in self.mod_mapping:
            name_meta = f"{name_mod}_path" if name_mod != "s2_mask" else "s2_mask"
            name_flair = self.mod_mapping[name_mod].upper()
            meta[name_meta] = (
                self.root_dir
                / f"{domain}_{name_flair}"
                / area
                / f"{domain}_{name_flair}_{area}_{pos}.tif"
            )

        s2_mtd = gpd.read_file(
            self.root_dir / f"{domain}_ALL_MTD" / f"{domain}_SENTINEL2_MTD_DATES.gpkg",
        )
        s2_mtd = s2_mtd.set_index("zone_id")
        dates_dict = s2_mtd.loc[f"{domain}_{area}", "acquisition_dates"]
        dates_dict = json.loads(dates_dict)
        meta["s2_dates"] = dict_datetimes(dates_dict, start=1)

        aerial_mtd = gpd.read_file(
            self.root_dir / f"{domain}_ALL_MTD" / f"{domain}_AERIAL_MTD_DATES.gpkg",
        )
        aerial_mtd = aerial_mtd.set_index("patch_id")
        dates_strs = [aerial_mtd.loc[f"{domain}_{area}_{pos}", "date"]]
        meta["aerial_dates"] = strs_datetimes(dates_strs, format_="%Y%m%d")

        aerial_date = meta["aerial_dates"]
        meta["dem_dates"] = aerial_date
        meta["flair_inc_lpis_dates"] = aerial_date

        inputs = self.preprocess_rasters(meta)

        inputs["ref_date"] = aerial_date

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.patch_ids)
