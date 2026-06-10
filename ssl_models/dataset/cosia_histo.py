"""COSIA-Histo dataset module."""

from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import strs_datetimes


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
        ssl_phase: Literal["pretrain", "probe", "finetune"],  # noqa: ARG002
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )

        self.root_dir = Path(root_dir)

        if dataset.gpkg_dir:
            self.gpkg_dir = self.root_dir / dataset.gpkg_dir
        else:
            self.gpkg_dir = self.root_dir

        modalities_path = self._resolve_gpkg_path(dataset.modalities_gpkg)
        split_path = self._resolve_gpkg_path(dataset.split_gpkg)

        modalities = gpd.read_file(modalities_path, engine="pyogrio", use_arrow=True)
        split = gpd.read_file(split_path, engine="pyogrio", use_arrow=True)

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

        # Accept common validation aliases used in split files.
        split_aliases = {
            "train": "train",
            "training": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "test": "test",
            "testing": "test",
        }
        normalized_split = (
            data[dataset.split_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(split_aliases)
        )
        self.data = data.loc[normalized_split == stage].reset_index(drop=True)
        self.base_length = len(self.data)
        self.repeats = dataset.repeats

        self.aerial_path_col = dataset.aerial_path_col
        self.target_path_col = dataset.target_path_col
        self.remap_to_others_after_class = dataset.remap_to_others_after_class
        self.cosia_missing = dataset.cosia.missing_val
        self.cosia_others_label = dataset.cosia.num_classes - 1
        self.default_dates = strs_datetimes(
            [dataset.default_date],
            fmt=dataset.default_date_fmt,
        )

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        idx, start_gcd = self.set_gcd_rng(
            idx,
            base_length=self.base_length,
            repeats=self.repeats,
        )

        row = self.data.iloc[idx]
        meta = {
            "aerial_path": self._resolve_raster_path(row[self.aerial_path_col]),
            "aerial_dates": self.default_dates,
            "cosia_path": self._resolve_raster_path(row[self.target_path_col]),
            "cosia_dates": self.default_dates,
        }

        inputs = self.preprocess_rasters(meta, start_gcd=start_gcd)

        if self.remap_to_others_after_class is not None and "cosia" in inputs:
            cosia = inputs["cosia"]
            valid_mask = cosia != self.cosia_missing
            remap_mask = valid_mask & (cosia >= self.remap_to_others_after_class)
            cosia[remap_mask] = self.cosia_others_label

        inputs["ref_date"] = meta["aerial_dates"]
        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2

    def _resolve_gpkg_path(self, gpkg_name: str) -> Path:
        path = Path(gpkg_name)
        if path.suffix != ".gpkg":
            path = path.with_suffix(".gpkg")
        return path if path.is_absolute() else self.gpkg_dir / path

    def _resolve_raster_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)

        if path.is_absolute():
            if not path.exists():
                msg = f"Raster path {path} does not exist."
                raise ValueError(msg)
            return path
        else:
            resolved_path = self.root_dir / path
            if not resolved_path.exists():
                resolved_path = self.gpkg_dir / path
                if not resolved_path.exists():
                    msg = (
                        f"Raster path {path} does not exist as absolute path, "
                        f"nor relative to root_dir {self.root_dir}, "
                        f"nor relative to gpkg_dir {self.gpkg_dir}."
                    )
                    raise ValueError(msg)
            return resolved_path

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