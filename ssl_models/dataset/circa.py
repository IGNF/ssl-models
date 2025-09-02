"""Circa data module."""

import json
from pathlib import Path
from typing import Literal

import numpy as np

from conf.dataset.utils import DatasetConfig
from ssl_models import RNG
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import filter_dates_ref, read_csv, strs_datetimes


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
            val_pretrain=dataset.val_pretrain,
            filter_percent=dataset.filter_percent,
        )
        self.stage = stage
        self.root_dir = root_dir
        self.test_dir = (
            "test_v2/aleatoire" if dataset.dataset_local else "optique_dataset"
        )

        self.repeats = dataset.repeats
        self.patch_ids = csv_data["patch_id"].tolist()
        if self.stage == "val":
            self.norm_min = csv_data["norm_min"].tolist()
            self.norm_max = csv_data["norm_max"].tolist()
        self.base_length = len(self.patch_ids)

        self.num_dates = {
            "s1_asc_zoom": dataset.s1_asc_zoom.num_dates,
            "s1_des_zoom": dataset.s1_des_zoom.num_dates,
            "s2_zoom": dataset.s2_zoom.num_dates,
        }

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        idx, start_gcd = self.sample_gcd(
            idx,
            base_length=self.base_length,
            repeats=self.repeats,
        )

        patch_id = self.patch_ids[idx]
        zone_id, row_col_id = patch_id.split("/")[1].split("_", maxsplit=1)
        zone_id = zone_id.split("-")[-1]

        meta = {}

        s1_dir = (
            self.root_dir
            / "radar_dataset_v4"
            / zone_id
            / f"MGRS25-{zone_id}_{row_col_id}"
        )
        meta["s1_asc_global_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_ASC.tif"
        )
        with Path(s1_dir / "dates_ASC.json").open() as f:
            meta["s1_asc_global_dates"] = strs_datetimes(json.load(f), fmt="%Y%m%d")

        meta["s1_asc_zoom_path"] = meta["s1_asc_global_path"]
        meta["s1_asc_zoom_dates"] = meta["s1_asc_global_dates"]

        meta["s1_des_global_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_DESC.tif"
        )
        with Path(s1_dir / "dates_DESC.json").open() as f:
            meta["s1_des_global_dates"] = strs_datetimes(json.load(f), fmt="%Y%m%d")

        meta["s1_des_zoom_path"] = meta["s1_des_global_path"]
        meta["s1_des_zoom_dates"] = meta["s1_des_global_dates"]

        s2_dir = (
            (self.root_dir / self.test_dir / zone_id / f"MGRS25-{zone_id}_{row_col_id}")
            if self.stage == "test"
            else (
                self.root_dir
                / "optique_dataset"
                / zone_id
                / f"MGRS25-{zone_id}_{row_col_id}"
            )
        )

        meta["s2_global_path"] = s2_dir / f"bands_stacked_{zone_id}_{row_col_id}.tif"
        meta["s2_global_mask"] = meta["s2_global_path"]
        with Path(s2_dir / "dates.json").open() as f:
            meta["s2_global_dates"] = strs_datetimes(json.load(f), fmt="%Y%m%d")

        meta["s2_zoom_path"] = meta["s2_global_path"]
        meta["s2_zoom_mask"] = meta["s2_global_path"]
        meta["s2_zoom_dates"] = meta["s2_global_dates"]

        ref_date = (
            meta["s2_zoom_dates"][len(meta["s2_zoom_dates"]) // 2]
            if self.stage != "test"
            else self.ref_dates[idx]
        )

        for name_mod, num_dates in self.num_dates.items():
            meta[f"{name_mod}_dates"], meta[f"{name_mod}_filter_dates"] = (
                filter_dates_ref(
                    meta[f"{name_mod}_dates"],
                    ref_date,
                    num_dates,
                )
            )
        meta["s2_global_filter_dates"] = np.array(
            [
                idx not in meta["s2_zoom_filter_dates"].tolist()
                for idx, _ in enumerate(meta["s2_global_dates"])
            ],
        )
        meta["s2_global_dates"] = meta["s2_global_dates"][
            meta["s2_global_filter_dates"]
        ]

        inputs = self.preprocess_rasters(
            meta,
            start_gcd=start_gcd,
        )
        for name_mod in (
            "s1_asc_global",
            "s1_des_global",
            "s1_asc_zoom",
            "s1_des_zoom",
        ):
            if name_mod not in inputs:
                continue
            inputs[name_mod] /= np.array([50.0, 50.0, 1.0, 1.0]).reshape([1, 4, 1, 1])

        inputs["ref_date"] = ref_date[None, :]
        
        if self.stage == "val":
            inputs["norm_min"] = self.norm_min[idx]
            inputs["norm_max"] = self.norm_max[idx]
            inputs["idx_date_ref"] = np.where((inputs["ref_date"] == inputs["s2_zoom_dates"]).all(axis=1))[0].item()

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2
