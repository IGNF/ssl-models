"""FLAIR dataset module."""

import json
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import dict_datetimes, strs_datetimes


class FLAIRDataset(GenericDataset):
    """FLAIR dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_dates: bool,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
        )

        csv_path = root_dir / f"{stage}.csv"
        csv_data = pd.read_csv(csv_path)

        self.root_dir = root_dir
        self.patch_ids = csv_data["patch_id"].tolist()

        self.mod_mapping = {
            "aerial": "aerial_rgbi",
            "aerial_rlt": "aerial-rlt_pan",
            "dem": "dem_elev",
            "spot": "spot_rgbi",
            "s2": "sentinel2_ts",
            "s2_mask": "sentinel2_msk-sc",
            "s1_asc": "sentinel1-asc_ts",
            "s1_des": "sentinel1-desc_ts",
            "flair_cosia": "aerial_label-cosia",
            "flair_lpis": "all_label-lpis",
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

        for name_mod in ("s2", "s1_asc", "s1_des"):
            name_flair = self.mod_mapping[name_mod].split("_")[0].upper()
            mtd_path = (
                self.root_dir
                / f"{domain}_ALL_MTD"
                / (f"{domain}_{name_flair}_MTD_DATES.gpkg")
            )
            mtd = gpd.read_file(mtd_path)
            mtd = mtd.set_index("zone_id")
            dates_dict = mtd.loc[f"{domain}_{area}", "acquisition_dates"]
            dates_dict = json.loads(dates_dict)
            meta[f"{name_mod}_dates"] = dict_datetimes(dates_dict, start=1)

        for name_mod in ("aerial", "spot"):
            name_flair = name_mod.upper()
            mtd_path = (
                self.root_dir
                / f"{domain}_ALL_MTD"
                / (f"{domain}_{name_flair}_MTD_DATES.gpkg")
            )
            mtd = gpd.read_file(mtd_path)
            mtd = mtd.set_index("patch_id")
            dates_strs = [mtd.loc[f"{domain}_{area}_{pos}", "date"]]
            meta[f"{name_mod}_dates"] = strs_datetimes(dates_strs, format_="%Y%m%d")

        aerial_date = meta["aerial_dates"]

        meta["dem_dates"] = aerial_date
        meta["flair_lpis_dates"] = aerial_date
        meta["flair_cosia_dates"] = aerial_date
        meta["flair_cosia_shift"] = -1

        inputs = self.preprocess_rasters(meta)

        inputs["ref_date"] = aerial_date

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.patch_ids)
