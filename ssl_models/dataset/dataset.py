"""Generic dataset module."""

import warnings
from abc import ABC
from math import gcd
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.osm_to_seg import process_geojson_file

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class GenericDataset(Dataset, ABC):
    """Generic dataset."""

    def __init__(
        self,
        use_transform: bool,
        dataset: DatasetConfig,
    ) -> None:
        self.rng = np.random.default_rng(seed=42)
        self.use_transform = use_transform
        self.dataset = dataset

    def preprocess_rasters(
        self,
        meta: dict[str, Path | str | np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Read raster inputs."""
        if any(
            f"{name_mod}_path" not in meta or f"{name_mod}_dates" not in meta
            for name_mod in self.dataset.rasters
        ):
            msg = "A modality is missing in meta data."
            raise ValueError(msg)

        sizes, sizes_date = {}, {}
        for name_mod, mod in self.dataset.rasters.items():
            sizes[name_mod] = round(self.dataset.total_meters / mod.resolution_meters)
            sizes_date[name_mod] = len(meta[f"{name_mod}_dates"])

        size_gcd = gcd(*sizes.values())
        crop_gcd = self.dataset.crop_meters / self.dataset.total_meters * size_gcd
        if not crop_gcd.is_integer():
            msg = (
                f"Crop meters does not correspond to an integer number of pixels."
                f"Use a multiple of {self.dataset.total_meters / size_gcd}."
            )
            raise ValueError(msg)

        crop_gcd = round(crop_gcd)
        start_gcd = self.rng.integers(
            low=0,
            high=size_gcd - crop_gcd + 1,
            size=2,
        )
        start, end = {}, {}
        for name_mod, mod in self.dataset.rasters.items():
            start_mod = start_gcd * (sizes[name_mod] // size_gcd)
            start_mod += meta.get(f"{name_mod}_shift", 0)  # align TreeSatAI-TS aerial
            start_mod_date = self.rng.integers(
                low=0,
                high=sizes_date[name_mod] % mod.num_dates + 1,
            )
            start[name_mod] = (*start_mod, start_mod_date)

            end_mod = (start_gcd + crop_gcd) * (sizes[name_mod] // size_gcd)
            end_mod_date = start_mod_date + mod.num_dates * (
                sizes_date[name_mod] // mod.num_dates
            )
            end[name_mod] = (*end_mod, end_mod_date)

        # read raster slices
        inputs = {}
        for name_mod, mod in self.dataset.rasters.items():
            input_mod, dates_mod = self.preprocess_raster(
                meta[f"{name_mod}_path"],
                meta[f"{name_mod}_dates"],
                meta.get(f"{name_mod}_mask", None),
                meta.get(f"{name_mod}_h5_name", None),
                meta.get(f"{name_mod}_h5_mask", None),
                start[name_mod],
                end[name_mod],
                **vars(mod),
            )
            inputs[name_mod] = input_mod
            inputs[f"{name_mod}_dates"] = dates_mod

        return inputs

    def preprocess_raster(  # noqa: PLR0915
        self,
        path_mod: Path,
        dates_mod: np.ndarray,
        mask: Path | None,
        h5_name: str | None,
        h5_mask: str | None,
        start: tuple[int],
        end: tuple[int],
        bands: int | list[list[int]],
        num_dates: int = 1,
        patch_dates: int | None = None,
        mask_threshold: float = 100.0,  # threshold on percentage cover
        norm_fac: float | None = None,
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
                    input_mod = input_mod.reshape(
                        (len(dates_mod), -1, *input_mod.shape[1:]),
                    )
                    input_mod = input_mod[:, bands]
                if use_mask and mask is not None:
                    with rasterio.open(mask) as src:
                        mask_mod = src.read(window=window)
                        mask_mod = mask_mod.reshape(
                            (len(dates_mod), -1, *mask_mod.shape[1:]),
                        )
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

        if len(dates_mod) != num_dates:
            input_mod = input_mod[slices[2]]
            input_mod = input_mod.reshape((num_dates, -1, *input_mod.shape[1:]))
            dates_mod = dates_mod[slices[2]]
            dates_mod = dates_mod.reshape((num_dates, -1, dates_mod.shape[1], 1, 1))
            if mask_mod is not None:
                mask_mod = mask_mod[slices[2]]
                mask_mod = mask_mod.reshape((num_dates, -1, *mask_mod.shape[1:]))
                mask_mod = (mask_mod > mask_threshold).any(axis=2, keepdims=True)
                mask_mod = mask_mod & ~mask_mod.all(axis=1, keepdims=True)
                input_mod = np.where(mask_mod, np.nan, input_mod)

            pan_mod = input_mod.sum(axis=2, keepdims=True)
            if patch_dates:
                pan_mod = self.rng.random(
                    (
                        *pan_mod.shape[:-2],
                        pan_mod.shape[-2] // patch_dates,
                        pan_mod.shape[-1] // patch_dates,
                    ),
                )
                pan_mod = np.repeat(pan_mod, repeats=patch_dates, axis=-2)
                pan_mod = np.repeat(pan_mod, repeats=patch_dates, axis=-1)

            mid_finite = np.isfinite(pan_mod).sum(axis=1, keepdims=True)
            mid_finite = np.floor_divide(mid_finite, 2)
            median_inds = np.argsort(pan_mod, axis=1)
            median_inds = np.take_along_axis(median_inds, mid_finite, axis=1)
            input_mod = np.take_along_axis(input_mod, median_inds, axis=1).squeeze(1)
            dates_mod = np.take_along_axis(dates_mod, median_inds, axis=1).squeeze(1)

        input_mod = input_mod.astype(np.float32)
        if norm_fac is not None:
            input_mod /= norm_fac
        return input_mod, dates_mod

    def transform_rasters(
        self,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Transform model rasters."""
        if self.use_transform:
            if self.rng.choice([True, False]):
                inputs.update(
                    {
                        name_mod: np.flip(inputs[name_mod], axis=2)
                        for name_mod in self.dataset.rasters
                    },
                )
            if self.rng.choice([True, False]):
                inputs.update(
                    {
                        name_mod: np.flip(inputs[name_mod], axis=3)
                        for name_mod in self.dataset.rasters
                    },
                )
            if self.rng.choice([True, False]):
                inputs.update(
                    {
                        name_mod: np.swapaxes(inputs[name_mod], 2, 3)
                        for name_mod in self.dataset.rasters
                    },
                )
            inputs.update(
                {
                    name_mod: np.ascontiguousarray(inputs[name_mod])
                    for name_mod in self.dataset.rasters
                },
            )
        return inputs
