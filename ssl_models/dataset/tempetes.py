"""Tempete dataset module."""

from pathlib import Path
from typing import Literal

import numpy as np

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import dict_datetimes, read_json, strs_datetimes


class TempetesDataset(GenericDataset):
    """Tempete dataset."""

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

        json_dir = Path(dataset.json_dir) if dataset.json_dir else root_dir
        self.json_data = read_json(
            json_dir=json_dir,
            stage=stage,
            ssl_phase=ssl_phase,
            version=dataset.version,
            balance_pretrain=dataset.balance_pretrain,
            val_pretrain=dataset.val_pretrain,
            test_pretrain=dataset.test_pretrain,
            filter_percent=dataset.filter_percent,
        )

        self.root_dir = root_dir
        self.patch_ids = self.json_data["id_patch"].tolist()
        self.base_length = len(self.patch_ids)
        self.repeats = dataset.repeats

        self.mod_mapping = {
            "pleiades": "pleiades_rgbi",
            "s2": "sentinel2_ts" ,
            "tempetes": "pleiades_label-tempete",
        }

        self.dates_str = {}

        self.dates_str["pleiades"] = {
            row["id_patch"]: row["PLEIADES_RGBI_DATES"]['1'] for _, row in self.json_data.iterrows()
        }

        self.dates_dict = {}

        self.dates_dict["s2"] = {
            "_".join(row["id_patch"].split("_")[:2]): row["SENTINEL2_TS_DATES"]
            for _, row in self.json_data.iterrows()
        }

        if dataset.version == "tempetes-only":
            self.label_ignore = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19)
        else:
            self.label_ignore = ()
        self.label_missing = dataset.tempetes.missing_val

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        idx, start_gcd = self.set_gcd_rng(
            idx,
            base_length=self.base_length,
            repeats=self.repeats,
        )

        patch_id = self.patch_ids[idx]
        zone_id = "_".join(patch_id.split("_")[:2])

        meta = {}
        for name_mod in self.mod_mapping:
            name_mapping = self.mod_mapping[name_mod].upper()
            meta[f"{name_mod}_path"] = (
                self.root_dir / self.json_data[f"{name_mapping}"][
                    self.json_data["id_patch"] == patch_id
                ].values[0]
            )

        meta["pleiades_dates"] = strs_datetimes(
            [self.dates_str["pleiades"][patch_id]],
            fmt="%Y%m%d",
        )
        meta["s2_dates"] = dict_datetimes(
            self.dates_dict["s2"][zone_id],
            start=1,
        )

        meta["tempetes_dates"] = meta["pleiades_dates"]

        inputs = self.preprocess_rasters(meta, start_gcd=start_gcd)

        inputs["tempetes"][
            np.isin(inputs["tempetes"], self.label_ignore)
        ] = self.label_missing

        inputs["ref_date"] = meta["tempetes_dates"]

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2