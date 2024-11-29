"""FLAIR dataset module."""

import json
from pathlib import Path

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
        root_dir: Path,
        csv_path: Path,
        use_transform: bool,
        dataset: DatasetConfig,
        *args,  # noqa: ARG002, ANN002
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(use_transform=use_transform, dataset=dataset)

        csv_data = pd.read_csv(csv_path)

        patch_ids = csv_data["patch_id"].tolist()

        self.root_dir = root_dir
        self.patch_ids = patch_ids

        self.mod_names_match = {
            "aerial": "aerial_rgbi",
            "aerial_rlt": "aerial-rlt_pan",
            "dem": "dem_elev",
            "spot": "spot_rgbi",
            "s2": "sentinel2_ts",
            "s1_asc": "sentinel1-asc_ts",
            "s1_desc": "sentinel1-desc_ts",
            "flair_cosia": "aerial_label-cosia",
            # "flair_lpis": "all_label-lpis"
        }

        self.mod_mtd_json = ["s1_asc", "s1_desc", "s2"]
        self.mod_mtd_gpkg = [
            "aerial",
            "aerial_rlt",
            "spot",
        ]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        (patch_domain, patch_area, patch_position) = self.patch_ids[idx].split("_")
        patch_metadata_dir = self.root_dir / f"{patch_domain}_ALL_MTD"

        meta = {}

        # tif images
        for mod, mod_name in self.mod_names_match.items():
            if mod == "aerial_rlt":
                meta[f"{mod}_path"] = (
                    self.root_dir
                    / f"{patch_domain.split('-')[0]}-195X_{mod_name.upper()}"
                    / patch_area
                    / f"{patch_domain}_{mod_name.upper()}_{patch_area}_{patch_position}.tif"
                )
            else:
                meta[f"{mod}_path"] = (
                    self.root_dir
                    / f"{patch_domain}_{mod_name.upper()}"
                    / patch_area
                    / f"{patch_domain}_{mod_name.upper()}_{patch_area}_{patch_position}.tif"
                )
        # mtd json
        for mod in self.mod_mtd_json:
            mtd_filename = (
                patch_metadata_dir
                / f"{patch_domain}_{self.mod_names_match[mod].split('_')[0].upper()}_MTD_DATES.json"
            )
            with open(mtd_filename) as file:
                dates_dict = json.load(file)[f"{patch_domain}_{patch_area}"]
            meta[f"{mod}_dates"] = dict_datetimes(dates_dict, start=1)

        # mtd gpkg
        for mod in self.mod_mtd_gpkg:
            if mod == "aerial_rlt":
                mtd_filename = (
                    patch_metadata_dir
                    / f"{patch_domain.split('-')[0]}-195X_{self.mod_names_match[mod].split('_')[0].upper()}_MTD_DATES.gpkg"
                )
            else:
                mtd_filename = (
                    patch_metadata_dir
                    / f"{patch_domain}_{self.mod_names_match[mod].split('_')[0].upper()}_MTD_DATES.gpkg"
                )
            mtd = gpd.read_file(mtd_filename)
            mtd = mtd.set_index("patch_id")
            if mod == "aerial_rlt":
                date_start = strs_datetimes(
                    [mtd.loc[
                        f"{patch_domain}_{patch_area}_{patch_position}",
                        "date_start",
                    ]],
                    format_="%Y%m%d"
                )
                date_end = strs_datetimes(
                    [mtd.loc[
                        f"{patch_domain}_{patch_area}_{patch_position}",
                        "date_end",
                    ]],
                    format_="%Y%m%d" 
                )
                meta[f"{mod}_dates"] = date_start + (date_end - date_start) / 2
            else:
                meta[f"{mod}_dates"] = strs_datetimes(
                    [mtd.loc[f"{patch_domain}_{patch_area}_{patch_position}", "date"]],
                    format_="%Y%m%d"
                )

        # meta["aerial_shift"] = 6
        meta["dem_dates"] = meta["aerial_dates"]
        meta["flair_cosia_dates"] = meta["aerial_dates"]
        # meta["all_label_lpis"] = None
        inputs = self.preprocess_rasters(meta)
        inputs["ref_date"] = meta["aerial_dates"]
        inputs["flair_cosia"] -= 1

        return inputs

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.patch_ids)
