"""FLAIR dataset module."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import dict_datetimes


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

        patch_domains = csv_data["domains"]
        patch_areas = csv_data["areas"]
        patch_positions = csv_data["position"]

        self.root_dir = root_dir
        self.patch_domains = patch_domains
        self.patch_areas = patch_areas
        self.patch_positions = patch_positions

        def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
            """Get dataset item."""
            patch_domain = self.patch_domains[idx]
            patch_area = self.patch_area[idx]
            patch_position = self.patch_positions[idx]

            patch_metadata_dir = self.root_dir / f"{patch_domain}_ALL_MTD"

            meta = {}
            modalities_tif = [
                "aerial_rgbi",
                # "aerial-rlt_pan",
                "sentinel1-asc_ts",
                "sentinel1-desc_ts",
                "sentinel2_ts",
                "dem_elev",
                "spot_rgbi",
                "aerial_label-cosia",
                # "all_label-lpis",
            ]
            for mod in modalities_tif:
                meta[f"{mod}_path"] = (
                    self.root_dir
                    / f"{patch_domain}_{mod.upper()}"
                    / patch_area
                    / f"{patch_domain}_{mod.upper()}_{patch_area}_{patch_position}.tif"
                )
            meta["aerial-rlt_pan_path"] = (
                self.root_dir
                / f"{patch_domain.split('-')[0]}-195X_{mod.upper()}"
                / patch_area
                / f"{patch_domain}_{mod.upper()}_{patch_area}_{patch_position}.tif"
            )

            modalities_mtd_json = ["sentinel1-asc", "sentinel1-desc", "sentinel2"]
            for mod in modalities_mtd_json:
                mtd_filename = (
                    patch_metadata_dir / f"{patch_domain}_{mod.upper()}_MTD_DATES.json"
                )
                with open(mtd_filename) as file:
                    dates_dict = json.load(file)[f"{patch_domain}_{patch_area}"]
                meta[f"{mod}_ts_dates"] = dict_datetimes(dates_dict)

            modalities_mtd_gpkg = [
                "aerial_rgbi",
                # "aerial-rlt_pan",  # TODO pas le même chemin, pas les même colonnes date, pas le même format
                "spot_rgbi",
            ]
            for mod in modalities_mtd_gpkg:
                if "rlt" in mod:
                    mtd_filename = (
                        patch_metadata_dir
                        / f"{patch_domain.split('-')[0]}-1950s_{mod.split('_')[0].upper()}_MTD_DATES.gpkg"
                    )
                else:
                    mtd_filename = (
                        patch_metadata_dir
                        / f"{patch_domain}_{mod.split('_')[0].upper()}_MTD_DATES.gpkg"
                    )
                mtd = gpd.read_file(mtd_filename)
                mtd = mtd.setindex("patch_id")
                meta[f"{mod}_dates"] = mtd.loc[
                    f"{patch_domain}_{patch_area}_{patch_position}", "date"
                ]  # TODO mettre au bon format 

                meta["dem_elev_dates"] = None
                meta["aerial_label-cosia_dates"] = None
                # meta["all_label_lpis"] = None
                inputs = self.preprocess_rasters(meta)
                inputs["ref_date"] = meta["aerial_rgbi_dates"]

        def __len__(self) -> int:
            """Return length of dataset."""
            return None  # TODO
