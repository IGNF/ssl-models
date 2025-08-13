"""Generic dataset module."""

import json
import warnings
from abc import ABC
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

from conf.dataset.utils import DatasetConfig, PatchSizeConfig
from ssl_models import RNG
from ssl_models.dataset.osm_to_seg import process_geojson_file
from ssl_models.dataset.utils import strs_datetimes

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class GenericDataset(Dataset, ABC):
    """Generic dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        model: Literal["mae", "mixmae"],
        random_dates: bool = False,
        random_crop: bool = False,
    ) -> None:
        self.stage = stage
        self.use_transform = use_transform
        self.dataset = dataset
        self.random_dates = random_dates and stage == "train"
        self.random_crop = random_crop and stage == "train"
        self.model = model

    @staticmethod
    def unflatten(x: np.ndarray, axis: int, reshape_axis: tuple) -> np.ndarray:
        """Unflatten specific axes in numpy array."""
        if axis == -1:
            axis = x.ndim - 1
        shape = (*x.shape[:axis], *reshape_axis, *x.shape[axis + 1 :])
        return x.reshape(shape)

    def sample_gcd(
        self,
        idx: int,
        base_length: int,
        repeats: int,
    ) -> tuple[int, np.ndarray]:
        """Sample crop part in image based on index of repetition."""
        if not self.random_crop:
            idx_repeat = idx // base_length
            idx_repeat = (idx_repeat // repeats, idx_repeat % repeats)
            start_gcd = np.array(
                (
                    idx_repeat[0] * self.dataset.size_gcd // repeats,
                    idx_repeat[1] * self.dataset.size_gcd // repeats,
                ),
            )
        else:
            start_gcd = None

        return (idx % base_length), start_gcd

    def crop_dates(
        self,
        tiles: list[str],
        root_dir: str,
        mod_dir: str,
        num_dates_zoom: int,
    ) -> pd.DataFrame:
        temporal_crop_df = pd.DataFrame(columns=["patch_id", "crop_idx", "ref_date"])
        for tile in tiles:
            dates_path = (
                root_dir / mod_dir / tile.rsplit("/", maxsplit=1)[0] / "dates.json"
            )
            with Path(dates_path).open() as file:
                dates = strs_datetimes(json.load(file), fmt="%Y%m%d")
            nb_crops = (len(dates)) // num_dates_zoom
            crops = [
                list(range(i * num_dates_zoom, (i + 1) * num_dates_zoom))
                for i in range(nb_crops - 1)
            ]
            crops.append(list(range((nb_crops - 1) * num_dates_zoom, len(dates))))
            date_ref = [
                dates[(len(crop) // 2) * (i + 1)] for i, crop in enumerate(crops)
            ]

            temporal_crop_df = pd.concat(
                [
                    temporal_crop_df,
                    pd.DataFrame(
                        {
                            "patch_id": [tile] * nb_crops,
                            "crop_idx": crops,
                            "ref_date": date_ref,
                        },
                    ),
                ],
            )
        return temporal_crop_df

    def average_pooling(self, x: np.ndarray, patch_size: int) -> np.ndarray:
        D, H, W = x.shape  # noqa: N806
        # mask partition : average pooling
        x = x.reshape(
            D,
            H // patch_size,
            patch_size,
            W // patch_size,
            patch_size,
        )
        return x.mean(axis=2).mean(axis=3)

    def max_pooling(self, x: np.ndarray, patch_size: int) -> np.ndarray:
        D, H, W = x.shape  # noqa: N806
        # mask partition : average pooling
        x = x.reshape(
            D,
            H // patch_size,
            patch_size,
            W // patch_size,
            patch_size,
        )
        return x.max(axis=2).max(axis=3)

    def preprocess_rasters(
        self,
        meta: dict[str, Path | str | np.ndarray],
        start_gcd: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Read raster inputs."""
        if any(
            f"{name_mod}_path" not in meta or f"{name_mod}_dates" not in meta
            for name_mod in self.dataset.rasters
        ):
            msg = "A modality is missing in meta data."
            raise ValueError(msg)

        sizes_date = {
            name_mod: len(meta[f"{name_mod}_dates"])
            for name_mod in self.dataset.rasters
        }
        if start_gcd is None:
            start_gcd = RNG.integers(
                low=0,
                high=self.dataset.size_gcd - self.dataset.crop_gcd + 1,
                size=2,
            )

        start, end = {}, {}
        for name_mod, mod in self.dataset.rasters.items():
            start_mod = start_gcd * (
                self.dataset.sizes[name_mod] // self.dataset.size_gcd
            )
            start_mod += meta.get(f"{name_mod}_shift", 0)  # align TreeSatAI-TS aerial
            start_mod_date = RNG.integers(
                low=0,
                high=sizes_date[name_mod] % mod.num_dates + 1,
            )
            start[name_mod] = (*start_mod, start_mod_date)

            end_mod = (start_gcd + self.dataset.crop_gcd) * (
                self.dataset.sizes[name_mod] // self.dataset.size_gcd
            )
            end_mod += meta.get(f"{name_mod}_shift", 0)  # align TreeSatAI-TS aerial
            end_mod_date = start_mod_date + mod.num_dates * (
                sizes_date[name_mod] // mod.num_dates
            )
            end[name_mod] = (*end_mod, end_mod_date)

        # read raster slices
        inputs = {}
        for name_mod, mod in self.dataset.rasters.items():
            input_mod, dates_mod, mask_mod, mask_mod_test = self.preprocess_raster(
                meta[f"{name_mod}_path"],
                meta[f"{name_mod}_dates"],
                meta.get(f"{name_mod}_mask", None),
                meta.get(f"{name_mod}_mask_bands", None),
                meta.get(f"{name_mod}_h5_name", None),
                meta.get(f"{name_mod}_h5_mask", None),
                meta.get(f"{name_mod}_window", None),
                meta.get(f"{name_mod}_temporal_crop", None),
                meta.get("ref_date"),
                start[name_mod],
                end[name_mod],
                **vars(mod),
            )
            inputs[name_mod] = input_mod
            inputs[f"{name_mod}_dates"] = dates_mod
            if meta.get(f"{name_mod}_need_mask", False):
                inputs[f"{name_mod}_mask"] = mask_mod
                if self.stage == "test" and meta.get(f"{name_mod}_window", False):
                    inputs[f"{name_mod}_mask_test"] = mask_mod_test

        return inputs

    def preprocess_raster(  # noqa: PLR0915, C901, PLR0913
        self,
        path_mod: Path,
        dates_mod: np.ndarray,
        mask: Path | None,
        mask_bands: list[list[int]],
        h5_name: str | None,
        h5_mask: str | None,
        zoom_window: bool,
        temporal_crop: list[int],
        ref_date: np.array,
        start: tuple[int],
        end: tuple[int],
        bands: int | list[list[int]],
        num_dates: int,
        mask_threshold: float,  # threshold on mask percentage cover
        norm_fac: float | None,
        log_scale: bool,
        offset: int | None,
        patch_size: PatchSizeConfig,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read raster modality."""
        slices = (
            slice(start[0], end[0]),
            slice(start[1], end[1]),
            slice(start[2], end[2]),
        )
        bands = (
            range(bands)
            if isinstance(bands, int)
            else [idx for band in bands for idx in band]
        )
        use_mask = (mask_threshold / 100.0) < 1.0
        mask_mod = None  # by default, no mask is used
        mask_mod_test = None

        match path_mod.suffix:
            case ".tif" | ".png" | ".jpg" | ".jpeg":
                with rasterio.open(path_mod) as src:
                    window = Window(
                        start[1],
                        start[0],
                        end[1] - start[1],
                        end[0] - start[0],
                    )
                    input_mod = src.read(window=window)
                    input_mod = self.unflatten(input_mod, 0, (len(dates_mod), -1))
                    input_mod = input_mod[:, bands]
                if use_mask and mask is not None:
                    bands = mask_bands if mask_bands else bands
                    with rasterio.open(mask) as src:
                        mask_mod = src.read(window=window)
                        mask_mod = self.unflatten(mask_mod, 0, (len(dates_mod), -1))
                        mask_mod = mask_mod[:, bands, :, :]
            case ".npy":
                npy_file = np.load(path_mod, mmap_mode="r")
                if npy_file.ndim < 4:  # noqa: PLR2004
                    npy_file = np.expand_dims(npy_file, axis=0)
                input_mod = npy_file[:, :, slices[0], slices[1]][:, bands]
                if use_mask and mask is not None:
                    mask_file = np.load(mask, mmap_mode="r")
                    mask_mod = mask_file[:, slices[0], slices[1]]
            case ".h5":
                with h5py.File(path_mod, "r") as h5_file:
                    input_mod = h5_file[h5_name][:, :, slices[0], slices[1]][:, bands]
                    if use_mask and h5_mask is not None:
                        mask_mod = h5_file[h5_mask][:, :, slices[0], slices[1]]
            case ".geojson":
                geojson_file = process_geojson_file(path_mod)
                geojson_file = np.expand_dims(geojson_file, axis=(0, 1))
                input_mod = geojson_file[:, :, slices[0], slices[1]]
            case _:
                msg = f"File format {path_mod.suffix} not supported."
                raise NotImplementedError(msg)

        patch_size = patch_size.mae if self.model == "mae" else patch_size.mixmae
        if zoom_window:
            if mask_mod is not None:
                mask_mod_test = mask_mod
                mask_mod = (
                    self.max_pooling(mask_mod, patch_size)
                    if self.stage == "test"
                    else self.average_pooling(mask_mod, patch_size)
                )
            num_dates = len(temporal_crop) if temporal_crop else num_dates
            zoom_dates_idx = (
                temporal_crop
                if temporal_crop
                else np.argsort(
                    [
                        abs(
                            (
                                datetime(int(x[0]), 1, 1)  # noqa: DTZ001
                                + timedelta(days=int(x[1]) - 1)
                                - (
                                    datetime(int(ref_date[0]), 1, 1)  # noqa: DTZ001
                                    + timedelta(days=int(ref_date[1]) - 1)
                                )
                            ).days,
                        )
                        for x in dates_mod
                    ],
                )[:num_dates]
            )
            zoom_dates_idx = np.sort(zoom_dates_idx)
            dates_mod = np.take(dates_mod, zoom_dates_idx, axis=0)
            dates_mod = np.expand_dims(dates_mod, axis=(2, 3))
            input_mod = np.take(input_mod, zoom_dates_idx, axis=0)
            if mask_mod is not None:
                mask_mod = np.take(mask_mod, zoom_dates_idx, axis=0)
                mask_mod = np.expand_dims(mask_mod, axis=1)
                mask_mod_test = np.take(mask_mod_test, zoom_dates_idx, axis=0)
                mask_mod_test = np.expand_dims(mask_mod_test, axis=1)

        elif len(dates_mod) != num_dates:
            input_mod = input_mod[slices[2]]
            input_mod = self.unflatten(input_mod, 0, (num_dates, -1))
            dates_mod = dates_mod[slices[2], :, None, None]
            dates_mod = self.unflatten(dates_mod, 0, (num_dates, -1))
            if mask_mod is not None:
                mask_mod = mask_mod[slices[2]]
                mask_mod = self.unflatten(mask_mod, 0, (num_dates, -1))
                mask_mod_criterion = (mask_mod > mask_threshold).any(
                    axis=2,
                    keepdims=True,
                )
                mask_mod_criterion = mask_mod_criterion & ~(
                    mask_mod_criterion.any(axis=(3, 4), keepdims=True).all(
                        axis=1,
                        keepdims=True,
                    )
                )
                input_mod = np.where(mask_mod_criterion, np.nan, input_mod)

            diff_mod = input_mod - np.nanmedian(input_mod, axis=1, keepdims=True)
            diff_mod = np.abs(diff_mod)
            if self.random_dates:
                diff_mod = 0 * diff_mod  # keep NaNs
                diff_mod += RNG.random(diff_mod.shape).astype(diff_mod.dtype)

            diff_mod = np.mean(diff_mod, axis=(2, 3, 4), keepdims=True)
            median_inds = np.nanargmin(diff_mod, axis=1, keepdims=True)
            input_mod = np.take_along_axis(input_mod, median_inds, axis=1).squeeze(1)
            dates_mod = np.take_along_axis(dates_mod, median_inds, axis=1).squeeze(1)
            if mask_mod is not None:
                mask_mod = np.take_along_axis(mask_mod, median_inds, axis=1).squeeze(1)
                mask_mod = mask_mod[:, 0, :, :]

                mask_mod_test = mask_mod
                mask_mod = self.average_pooling(mask_mod, patch_size)
                mask_mod = np.expand_dims(mask_mod, axis=(1))
                mask_mod_test = np.expand_dims(mask_mod_test, axis=(1))

        input_mod = input_mod.astype(np.float32)

        if offset is not None:
            input_mod += offset

        if log_scale:
            input_mod = np.log(np.maximum(input_mod, 1e-10))

        if norm_fac is not None:
            input_mod /= norm_fac

        return input_mod, dates_mod, mask_mod, mask_mod_test

    def transform_rasters(
        self,
        inputs: dict[str, np.ndarray],
        additional_inputs: dict[str, None] = {},  # noqa: B006
    ) -> dict[str, np.ndarray]:
        """Transform model rasters."""
        if self.use_transform:
            if RNG.choice([True, False]):
                inputs.update(
                    {
                        name_mod: np.flip(inputs[name_mod], axis=2)
                        for name_mod in self.dataset.rasters
                        | self.dataset.masks
                        | additional_inputs
                    },
                )
            if RNG.choice([True, False]):
                inputs.update(
                    {
                        name_mod: np.flip(inputs[name_mod], axis=3)
                        for name_mod in self.dataset.rasters
                        | self.dataset.masks
                        | additional_inputs
                    },
                )
            if RNG.choice([True, False]):
                inputs.update(
                    {
                        name_mod: np.swapaxes(inputs[name_mod], 2, 3)
                        for name_mod in self.dataset.rasters
                        | self.dataset.masks
                        | additional_inputs
                    },
                )
            inputs.update(
                {
                    name_mod: np.ascontiguousarray(inputs[name_mod])
                    for name_mod in self.dataset.rasters
                    | self.dataset.masks
                    | additional_inputs
                },
            )
        return inputs
