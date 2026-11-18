"""COSIA-Histo dataset module."""

from copy import copy
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import strs_datetimes

# Accepted spellings of the splits in the split table.
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


class COSIAHistoDataset(GenericDataset):
    """COSIA-Histo dataset based on modality and split GeoPackage tables."""

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
        self.ssl_phase = ssl_phase

        if self.ssl_phase == "pretrain":
            # Targets are unused in pretraining and may be missing for some patches.
            self.dataset = copy(dataset)
            self.dataset.rasters = {
                name_mod: mod
                for name_mod, mod in dataset.rasters.items()
                if name_mod not in dataset.targets
            }

        self.root_dir = Path(root_dir)
        self.gpkg_dir = (
            self.root_dir / dataset.gpkg_dir if dataset.gpkg_dir else self.root_dir
        )

        modalities = gpd.read_file(
            self._resolve_gpkg_path(dataset.modalities_gpkg),
            engine="pyogrio",
            use_arrow=True,
        )
        split = gpd.read_file(
            self._resolve_gpkg_path(dataset.split_gpkg),
            engine="pyogrio",
            use_arrow=True,
        )

        self._check_required_cols(
            modalities,
            [dataset.fid_col, dataset.aerial_path_col, dataset.target_path_col],
            table_name="modalities",
        )
        self._check_required_cols(
            split,
            [dataset.fid_col, dataset.split_col],
            table_name="split",
        )

        data = modalities.merge(
            split[[dataset.fid_col, dataset.split_col]],
            on=dataset.fid_col,
            how="inner",
        )
        splits = data[dataset.split_col].astype(str).str.strip().str.lower()
        data = data.loc[splits.replace(SPLIT_ALIASES) == stage]

        self.aerial_paths = data[dataset.aerial_path_col].tolist()
        self.target_paths = data[dataset.target_path_col].tolist()
        self.base_length = len(data)
        self.repeats = dataset.repeats

        self.default_dates = strs_datetimes(
            [dataset.default_date],
            fmt=dataset.default_date_fmt,
        )

        self.label_missing = dataset.cosia.missing_val
        self.class_remap_luts = {
            name_target: np.array(remap, dtype=np.int64)
            for name_target, remap in dataset.class_remaps.items()
            if name_target in dataset.targets
        }

    def build_meta(self, idx: int) -> dict:
        """Return paths and dates of each modality.

        Args:
        ----
            idx: Index of the sample.

        Returns:
        -------
            A dictionary containing paths and dates of each modality.

        """
        meta = {
            "aerial_path": self._resolve_raster_path(self.aerial_paths[idx]),
            "aerial_dates": self.default_dates,
        }
        target_path = self.target_paths[idx]
        # Unannotated patches have no label path.
        if self.ssl_phase != "pretrain" and isinstance(target_path, str):
            target_path = self._resolve_raster_path(target_path)
            for name_target in self.dataset.targets:
                meta[f"{name_target}_path"] = target_path
                meta[f"{name_target}_dates"] = self.default_dates

        return meta

    def postprocess(
        self,
        inputs: dict[str, np.ndarray],
        meta: dict,
        idx: int,  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Remap classes and set reference date."""
        for name_target, lut in self.class_remap_luts.items():
            if name_target not in inputs:
                continue
            mask = inputs[name_target]
            valid = mask != self.label_missing
            mask[valid] = lut[mask[valid].astype(np.int64)]

        inputs["ref_date"] = meta["aerial_dates"]

        return inputs

    def _resolve_gpkg_path(self, gpkg_name: str) -> Path:
        path = Path(gpkg_name).with_suffix(".gpkg")
        return path if path.is_absolute() else self.gpkg_dir / path

    def _resolve_raster_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [self.root_dir / path, self.gpkg_dir / path]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        msg = f"Raster path {path} not found. Tried: {[str(c) for c in candidates]}."
        raise ValueError(msg)

    @staticmethod
    def _check_required_cols(
        table: gpd.GeoDataFrame,
        required_cols: list[str],
        table_name: str,
    ) -> None:
        missing_cols = [col for col in required_cols if col not in table.columns]
        if missing_cols:
            msg = f"Missing columns {missing_cols} in {table_name} table."
            raise ValueError(msg)
