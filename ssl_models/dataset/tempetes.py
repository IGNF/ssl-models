"""Tempetes dataset module."""

from copy import copy
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pyogrio import read_dataframe

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import dict_datetimes, read_csv

# Metadata column holding the path and the dates of each modality.
MOD_COLUMNS = {
    "pleiades": ("PLEIADES_RGBI", "PLEIADES_RGBI_DATES"),
    "s2": ("SENTINEL2_TS", "SENTINEL2_TS_DATES"),
    "tempetes": ("PLEIADES_LABEL-TEMPETE", None),
}


def parse_dates(dates: pd.Series) -> list[np.ndarray]:
    """Convert json date strings, parsing each distinct string once."""
    parsed = {date: dict_datetimes(date, start=1) for date in dates.unique()}
    return [parsed[date] for date in dates]


class TempetesDataset(GenericDataset):
    """Tempetes dataset."""

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

        if ssl_phase == "pretrain":
            # Targets are unused in pretraining and missing for most patches.
            self.dataset = copy(dataset)
            self.dataset.rasters = {
                name_mod: mod
                for name_mod, mod in dataset.rasters.items()
                if name_mod not in dataset.targets
            }

        csv_dir = Path(dataset.csv_dir) if dataset.csv_dir else root_dir
        csv_data = read_csv(
            csv_dir=csv_dir,
            stage=stage,
            ssl_phase=ssl_phase,
            version="pretrain" if ssl_phase == "pretrain" else "finetune",
            balance_pretrain=dataset.balance_pretrain,
            val_pretrain=dataset.val_pretrain,
            test_pretrain=dataset.test_pretrain,
            filter_percent=dataset.filter_percent,
            fold=dataset.fold,
        )

        self.root_dir = root_dir
        self.patch_ids = csv_data["id_patch"].tolist()
        self.base_length = len(self.patch_ids)
        self.repeats = dataset.repeats

        meta_path = csv_dir / f"global_metadata_{dataset.total_meters:g}m.gpkg"
        columns = [col for cols in MOD_COLUMNS.values() for col in cols if col]
        meta_data = read_dataframe(
            meta_path,
            columns=["id_patch", *columns],
            read_geometry=False,
            use_arrow=True,
        )
        meta_data = meta_data.set_index("id_patch").loc[self.patch_ids]

        self.paths = {
            name_mod: meta_data[col_path].tolist()
            for name_mod, (col_path, _) in MOD_COLUMNS.items()
        }
        self.dates = {
            name_mod: parse_dates(meta_data[col_dates])
            for name_mod, (_, col_dates) in MOD_COLUMNS.items()
            if col_dates
        }

        self.label_missing = dataset.tempetes.missing_val

        self.class_remap_lut = None
        if dataset.class_remap is not None:
            lut = np.arange(dataset._original_num_classes, dtype=np.int64)  # noqa: SLF001
            for src, dst in dataset.class_remap.items():
                lut[src] = dst
            self.class_remap_lut = lut

    def build_meta(self, idx: int) -> dict:
        """Return paths and dates of each modality."""
        meta = {}
        for name_mod, paths in self.paths.items():
            if isinstance(paths[idx], str):  # unlabeled patches have no label path
                meta[f"{name_mod}_path"] = self.root_dir / paths[idx]

        meta["pleiades_dates"] = self.dates["pleiades"][idx]
        meta["s2_dates"] = self.dates["s2"][idx]
        meta["tempetes_dates"] = meta["pleiades_dates"]

        return meta

    def postprocess(
        self,
        inputs: dict[str, np.ndarray],
        meta: dict,
        idx: int,  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Remap classes and set reference date."""
        if self.class_remap_lut is not None and "tempetes" in inputs:
            mask = inputs["tempetes"]
            valid = mask != self.label_missing
            mask[valid] = self.class_remap_lut[mask[valid].astype(np.int64)]

        inputs["ref_date"] = meta["tempetes_dates"]

        return inputs
