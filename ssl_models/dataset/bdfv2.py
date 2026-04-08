"""BDFV2 dataset module."""

from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import read_csv, strs_datetimes


class BDFV2Dataset(GenericDataset):
    """BDFV2 dataset."""

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

        csv_dir = Path(dataset.csv_dir) if dataset.csv_dir else root_dir
        csv_data = read_csv(
            csv_dir=csv_dir,
            stage=stage,
            ssl_phase=ssl_phase,
            val_pretrain=dataset.val_pretrain,
            test_pretrain=dataset.test_pretrain,
        )

        self.root_dir = root_dir
        self.patch_ids = csv_data["patch_id"].tolist()
        self.base_length = len(self.patch_ids)
        self.repeats = dataset.repeats

        self.mod_mapping = {
            "aerial": "aerial_rgbi",
            "dem": "dem_elev",
            "s2": "sentinel2-3a_ts",
            "s2_mask": "sentinel2-3a_msk-sc",
        }

        self.dates_str = {}
        for name_mod in ("aerial",):
            name_flair = self.mod_mapping[name_mod].split("_")[0].upper()
            dates_path = (
                self.root_dir / "GLOBAL_ALL_MTD" / f"GLOBAL_{name_flair}_MTD_DATES.gpkg"
            )
            dates_mod = gpd.read_file(
                dates_path,
                engine="pyogrio",
                ignore_geometry=True,
                use_arrow=True,
            )
            self.dates_str[name_mod] = {
                row["patch_id"]: row["date"] for _, row in dates_mod.iterrows()
            }

        tgt_path = self.root_dir / "GLOBAL_ALL_MTD" / "GLOBAL_ALL_MTD_LABEL-BDFV2.gpkg"
        target = gpd.read_file(
            tgt_path,
            engine="pyogrio",
            ignore_geometry=True,
            use_arrow=True,
        )

        cls_names, counts = np.unique(target["main_tfv"], return_counts=True)
        cls_idx = {
            cls_name: cls_idx
            for cls_idx, (_, cls_name) in enumerate(
                sorted(zip(counts, cls_names), reverse=True),
            )
        }
        target["cls_idx"] = target["main_tfv"].map(cls_idx)

        target = target[["patch_id", "cls_idx"]].set_index("patch_id")
        self.target = target.loc[self.patch_ids]["cls_idx"].tolist()

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        idx, start_gcd = self.set_gcd_rng(
            idx,
            base_length=self.base_length,
            repeats=self.repeats,
        )

        patch_id = self.patch_ids[idx]
        domain, pos = patch_id.split("_")
        year = int(domain.split("-")[-1])

        meta = {}
        for name_mod in self.mod_mapping:
            name_bdfv2 = self.mod_mapping[name_mod].upper()
            meta[f"{name_mod}_path" if name_mod != "s2_mask" else "s2_mask"] = (
                self.root_dir
                / f"{domain}_{name_bdfv2}"
                / f"{domain}_{name_bdfv2}_{pos}.tif"
            )

        for name_mod in ("aerial",):
            meta[f"{name_mod}_dates"] = strs_datetimes(
                [self.dates_str[name_mod][patch_id]],
                fmt="%Y%m%d",
            )
        for name_mod in ("s2",):
            meta[f"{name_mod}_dates"] = strs_datetimes(
                [f"{year}{month}15" for month in range(1, 13)],
                fmt="%Y%m%d",
            )
        meta["dem_dates"] = meta["aerial_dates"]

        inputs = self.preprocess_rasters(meta, start_gcd=start_gcd)

        inputs["bdfv2"] = np.array(self.target[idx])
        inputs["ref_date"] = meta["aerial_dates"]

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2
