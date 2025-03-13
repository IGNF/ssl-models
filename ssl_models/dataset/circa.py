"""Circa data module."""

import json
from pathlib import Path
from typing import Literal
import random
import numpy as np

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import read_csv, strs_datetimes


class CircaDataset(GenericDataset):
    """Circa dataset."""

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
            version=dataset.version,
            balance_pretrain=dataset.balance_pretrain,
            val_pretrain=dataset.val_pretrain,
            filter_percent=dataset.filter_percent,
        )

        self.root_dir = root_dir
        self.patch_ids = csv_data["patch_id"].tolist()
        self.base_length = len(self.patch_ids)
        self.repeats = dataset.repeats

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        idx, start_gcd = self.sample_gcd(
            idx,
            base_length=self.base_length,
            repeats=self.repeats,
        )

        patch_id = self.patch_ids[idx]
        zone_id, row_col_id = patch_id.split("/")[2].split("_", maxsplit=1)
        zone_id = zone_id.split("-")[-1]

        meta = {}

        s1_dir = (
            self.root_dir / "radar_dataset_v4"
            / zone_id / f"MGRS25-{zone_id}_{row_col_id}"
        )
        meta["s1_asc_global_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_ASC.tif"
        )
        meta["s1_asc_zoom_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_ASC.tif"
        )
        with open(s1_dir / "dates_ASC.json") as f:
            meta["s1_asc_global_dates"] = strs_datetimes(json.load(f), fmt="%Y%m%d")
            meta["s1_asc_zoom_dates"] = meta["s1_asc_global_dates"]

        meta["s1_des_global_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_DESC.tif"
        )
        meta["s1_des_zoom_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_DESC.tif"
        )
        with open(s1_dir / "dates_DESC.json") as f:
            meta["s1_des_global_dates"] = strs_datetimes(json.load(f), fmt="%Y%m%d")
            meta["s1_des_zoom_dates"] = meta["s1_des_global_dates"]

        s2_dir = (
            self.root_dir / "optique_dataset"
            / zone_id / f"MGRS25-{zone_id}_{row_col_id}"
        )
        meta["s2_global_path"] = s2_dir / f"bands_stacked_{zone_id}_{row_col_id}.tif"
        meta["s2_zoom_path"] = s2_dir / f"bands_stacked_{zone_id}_{row_col_id}.tif"
        with open(s2_dir / "dates.json") as f:
            meta["s2_global_dates"] = strs_datetimes(json.load(f), fmt="%Y%m%d")
            meta["s2_zoom_dates"] = meta["s2_global_dates"]

        meta["s2_cloud_mask_global_path"] = (
            s2_dir / f"bands_stacked_{zone_id}_{row_col_id}.tif"
        )
        meta["s2_cloud_mask_zoom_path"] = (
            s2_dir / f"bands_stacked_{zone_id}_{row_col_id}.tif"
        )
        meta["s2_cloud_mask_global_dates"] = meta["s2_global_dates"]
        meta["s2_cloud_mask_zoom_dates"] = meta["s2_zoom_dates"]

        ref_date = random.choice(meta["s1_asc_zoom_dates"])
        meta["ref_date"] = ref_date

        inputs = self.preprocess_rasters(meta, start_gcd=start_gcd)
        inputs["ref_date"] = np.expand_dims(ref_date, axis=0)

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2
