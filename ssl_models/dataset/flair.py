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

        deps = {x.split("_", maxsplit=1)[0] for x in patch_ids}
        dpkg_dep_dates = {}
        for dep in deps:
            dpkg_dep_dates[dep] = {}
            for mod in ["aerial", "spot"]:
                dpkg_dep_dates[dep][mod] = gpd.read_file(
                    root_dir / f"{dep}_ALL_MTD" / f"{dep}_{mod.upper()}_MTD_DATES.gpkg",
                    columns=["patch_id", "date"],
                    ignore_geometry=True,
                )

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
            "flair_lpis": "all_label-lpis",
        }

        self.mod_mtd_json = ["s1_asc", "s1_desc", "s2"]
        self.dep_mtd_dpkg = dpkg_dep_dates

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        (patch_domain, patch_area, patch_position) = self.patch_ids[idx].split("_")
        patch_metadata_dir = self.root_dir / f"{patch_domain}_ALL_MTD"

        meta = {}

        # tif images
        for mod, mod_name in self.mod_names_match.items():
            meta[f"{mod}_path"] = (
                self.root_dir
                / f"{patch_domain}_{mod_name.upper()}"
                / patch_area
                / (
                    f"{patch_domain}_{mod_name.upper()}"
                    f"_{patch_area}_{patch_position}.tif"
                )
            )

        # mtd json
        for mod in self.mod_mtd_json:
            mtd_filename = patch_metadata_dir / (
                f"{patch_domain}_"
                f"{self.mod_names_match[mod].split('_')[0].upper()}_MTD_DATES.json"
            )
            with mtd_filename.open() as file:
                dates_dict = json.load(file)[f"{patch_domain}_{patch_area}"]
            meta[f"{mod}_dates"] = dict_datetimes(dates_dict, start=1)

        # mtd gpkg
        mtd_aerial = self.dep_mtd_dpkg[patch_domain]["aerial"].set_index("patch_id")
        mtd_spot = self.dep_mtd_dpkg[patch_domain]["spot"].set_index("patch_id")
        meta["aerial_dates"] = strs_datetimes(
            [mtd_aerial.loc[f"{patch_domain}_{patch_area}_{patch_position}", "date"]],
            format_="%Y%m%d",
        )
        meta["spot_dates"] = strs_datetimes(
            [mtd_spot.loc[f"{patch_domain}_{patch_area}_{patch_position}", "date"]],
            format_="%Y%m%d",
        )

        meta["dem_dates"] = meta["aerial_dates"]
        for target in self.dataset.targets:
            meta[f"{target}_dates"] = meta["aerial_dates"]

        meta["s2_mask"] = (
            self.root_dir
            / f"{patch_domain}_SENTINEL2_MSK-SC"
            / patch_area
            / f"{patch_domain}_SENTINEL2_MSK-SC_{patch_area}_{patch_position}.tif"
        )
        meta["s2_stacked_dates"] = True
        meta["s1_asc_stacked_dates"] = True
        meta["s1_desc_stacked_dates"] = True

        inputs = self.preprocess_rasters(meta)
        inputs["ref_date"] = meta["aerial_dates"]

        if "flair_cosia" in inputs:
            inputs["flair_cosia"] -= 1

        return inputs

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.patch_ids)
