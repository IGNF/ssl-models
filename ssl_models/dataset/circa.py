"""Circa data module."""

import json
from pathlib import Path
from typing import Literal

import numpy as np

from conf.dataset.utils import DatasetConfig
from conf.mask import MaskConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import (
    cloud_mae_masking,
    cloud_mae_masking_test,
    mask_local_global_consistency,
    read_csv,
    strs_datetimes,
)


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
        model: Literal["mae", "mixmae"],
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
            model=model,
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
            nrows=1,
        )
        self.stage = stage
        self.root_dir = root_dir
        self.patch_ids = csv_data["patch_id"].tolist()
        self.base_length = len(self.patch_ids)
        self.repeats = dataset.repeats
        self.dataset_masks = dataset.masks
        self.dataset_inputs = dataset.inputs
        self.rng = np.random.default_rng(seed=42)
        self.mask_ratio = MaskConfig.mask_ratio
        self.mae_mask_value = MaskConfig.mae_mask_value
        self.mae_mask_ratio = 0.1

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
            self.root_dir
            / "radar_dataset_v4"
            / zone_id
            / f"MGRS25-{zone_id}_{row_col_id}"
        )
        meta["s1_asc_global_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_ASC.tif"
        )
        meta["s1_asc_zoom_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_ASC.tif"
        )
        with Path(s1_dir / "dates_ASC.json").open() as f:
            meta["s1_asc_global_dates"] = strs_datetimes(json.load(f), fmt="%Y%m%d")
            meta["s1_asc_zoom_dates"] = meta["s1_asc_global_dates"]
            meta["s1_asc_zoom_window"] = True

        meta["s1_des_global_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_DESC.tif"
        )
        meta["s1_des_zoom_path"] = (
            s1_dir / f"bands_stacked_MGRS25-{zone_id}_{row_col_id}_DESC.tif"
        )
        with Path(s1_dir / "dates_DESC.json").open() as f:
            meta["s1_des_global_dates"] = strs_datetimes(json.load(f), fmt="%Y%m%d")
            meta["s1_des_zoom_dates"] = meta["s1_des_global_dates"]
            meta["s1_des_zoom_window"] = True

        if self.stage == "test":
            s2_dir = (
                self.root_dir
                / "test"
                / "aleatoire"
                / zone_id
                / f"MGRS25-{zone_id}_{row_col_id}"
            )
        else:
            s2_dir = (
                self.root_dir
                / "optique_dataset"
                / zone_id
                / f"MGRS25-{zone_id}_{row_col_id}"
            )
        meta["s2_global_path"] = s2_dir / f"bands_stacked_{zone_id}_{row_col_id}.tif"
        meta["s2_zoom_path"] = s2_dir / f"bands_stacked_{zone_id}_{row_col_id}.tif"
        meta["s2_zoom_mask"] = meta["s2_zoom_path"]
        meta["s2_zoom_mask_bands"] = 10
        meta["s2_zoom_need_mask"] = True
        meta["s2_zoom_window"] = True
        with Path(s2_dir / "dates.json").open() as f:
            meta["s2_global_dates"] = strs_datetimes(json.load(f), fmt="%Y%m%d")
            meta["s2_zoom_dates"] = meta["s2_global_dates"]
            meta["s2_zoom_window"] = True
        meta["s2_global_mask"] = meta["s2_global_path"]
        meta["s2_global_mask_bands"] = [10, 11]
        meta["s2_global_need_mask"] = True

        ref_date = self.rng.choice(meta["s1_asc_zoom_dates"])
        meta["ref_date"] = ref_date

        inputs = self.preprocess_rasters(meta, start_gcd=start_gcd)
        inputs["ref_date"] = np.expand_dims(ref_date, axis=0)

        # mask clouds in local window
        if self.stage == "test":
            inputs["s2_zoom_mask"] = cloud_mae_masking_test(
                inputs["s2_zoom_mask"],
                self.mask_ratio,
                mae_masking_ratio=0.1,
                mask_value=self.mae_mask_value,
            )
            inputs["s2_zoom_mask_test"][inputs["s2_zoom_mask_test"] == 1] = 0
            inputs["s2_zoom_mask_test"][inputs["s2_zoom_mask_test"] == self.mae_mask_value] = 1
        else:
            inputs["s2_zoom_mask"] = cloud_mae_masking(
                inputs["s2_zoom_mask"],
                self.dataset_masks["s2_zoom_mask"].mask_threshold,
                self.mask_ratio,
                self.mae_mask_value,
                mae_masking_ratio=self.mae_mask_ratio,
            )

        # # corresponding mask in global window
        # inputs["s2_global_mask"] = mask_local_global_consistency(
        #     inputs["s2_zoom_dates"],
        #     inputs["s2_zoom_mask"],
        #     inputs["s2_global_dates"],
        #     inputs["s2_global_mask"],
        #     self.mae_mask_value,
        #     self.dataset_masks["s2_global_mask"].mask_threshold,
        #     mask_ratio=self.mask_ratio + self.mae_mask_ratio,
        # )

        if self.stage == "test":
            inputs["s2_zoom_mask_test"] = np.repeat(
                inputs["s2_zoom_mask_test"],
                self.dataset.inputs["s2_zoom"].bands,
                axis=1
            ).astype(bool)
            return self.transform_rasters(inputs, {"s2_zoom_mask_test": None})
        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2
