"""Generic dataset module."""

from abc import ABC
from pathlib import Path

import albumentations as A  # noqa: N812
import h5py
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

from conf.dataset.utils import RASTER_TARGETS, DatasetConfig


class GenericDataset(Dataset, ABC):
    """Generic dataset."""

    def __init__(
        self,
        transform: A.Compose | None,
        dataset: DatasetConfig,
    ) -> None:
        self.rng = np.random.default_rng()
        self.transform = transform

        self.filter_rasters = dataset.filter_inputs + [
            name_mod
            for name_mod, mod in dataset.targets.items()
            if mod.type_target in RASTER_TARGETS
        ]
        self.ground_meters = dataset.ground_meters
        self.crop_meters = dataset.crop_meters

    def preprocess_rasters(
        self,
        meta: dict[str, Path | str | np.ndarray],
    ) -> dict[str, np.ndarray | list]:
        """Read raster inputs."""
        for name_mod in self.mods_rasters:
            if (
                name_mod not in meta[f"{name_mod}_path"]
                or name_mod not in meta[f"{name_mod}_dates"]
            ):
                msg = ""
                raise ValueError(msg)

        sizes, sizes_date = {}, {}
        for name_mod, mod in self.mods_raster.items():
            sizes[name_mod] = self.ground_meters / mod.resolution_meters
            sizes_date[name_mod] = len(meta[f"{name_mod}_dates"])

        size_min = min(sizes.values())
        crop_min = (self.crop_meters / self.ground_meters) * size_min
        start_min = self.rng.integers(
            low=0,
            high=size_min - round(crop_min) + 1,
            size=2,
        )
        if not crop_min.is_integer():
            msg = "Crop range does not correspond to an integer number of pixels."
            raise ValueError(msg)

        for name_mod, size in self.sizes.items():
            if size % size_min != 0:
                msg = f"Size of raster {name_mod} not a multiple of size min."
                raise ValueError(msg)

        start, end = {}, {}
        for name_mod, mod in self.mods_raster.items():
            start_mod = start_min * (sizes[name_mod] // size_min)
            start_mod_date = self.rng.integers(
                low=0,
                high=sizes_date[name_mod] % mod.num_dates,
            )
            start[name_mod] = tuple(*start_mod, start_mod_date)

            end_mod = start_mod + crop_min * (sizes[name_mod] // size_min)
            end_mod_date = start_mod_date + mod.num_dates * (
                sizes_date[name_mod] // mod.num_dates
            )
            end[name_mod] = tuple(*end_mod, end_mod_date)

        inputs = {}
        for name_mod in self.mods_raster:
            input_mod, dates_mod = self.preprocess_raster(
                meta[f"{name_mod}_path"],
                meta[f"{name_mod}_dates"],
                meta.get(f"{name_mod}_mask", None),
                meta.get(f"{name_mod}_h5_name", None),
                meta.get(f"{name_mod}_h5_mask", None),
                start,
                end,
                **vars(mod),
            )
            inputs[name_mod] = input_mod
            inputs[f"{name_mod}_dates"] = dates_mod

        return inputs

    def preprocess_raster(
        self,
        path: Path,
        dates: np.ndarray,
        mask: Path | None,
        h5_name: str | None,
        h5_mask: str | None,
        start: tuple[int],
        end: tuple[int],
        bands: int | list[int],
        num_dates: int = 1,
        use_mask: bool = False,
        norm_fac: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read raster modality."""
        slices = (
            slice(start[0], end[0]),
            slice(start[1], end[1]),
            slice(start[2], end[2]),
        )
        mask_mod = None  # by default, not mask is used

        match path.suffix:
            case ".tif" | ".png":
                with rasterio.open(path) as src:
                    bands = bands if isinstance(bands, list) else range(src.count)
                    window = Window(start[0], start[1], end[0], end[1])
                    input_mod = src.read(bands, window=window)
                    input_mod = input_mod.reshape((len(dates), -1, *input_mod[1:]))
            case ".npy":
                npy_file = np.load(path, mmap_mode="r")
                bands = (
                    bands if isinstance(bands, list) else slice(0, npy_file.shape[1])
                )
                input_mod = npy_file[:, bands, slices[0], slices[1]]
                if use_mask and mask is not None:
                    mask_file = np.load(mask, mmap_mode="r")
                    mask_mod = mask_file[:, :, slices[0], slices[1]]
            case ".h5":
                with h5py.File(path, "r") as h5_file:
                    bands = (
                        bands
                        if isinstance(bands, list)
                        else slice(0, h5_file[h5_name].shape[1])
                    )
                    input_mod = h5_file[h5_name][:, bands, slices[0], slices[1]]
                    if use_mask and h5_mask is not None:
                        mask_mod = h5_file[h5_mask][:, :, slices[0], slices[1]]
            case _:
                msg = f"File format {path.suffix} not supported."
                raise NotImplementedError(msg)

        if mask_mod is not None:
            mask_mod = mask_mod.any(axis=-3, keepdims=True)  # either cloud or snow mask
            mask_mod = mask_mod & ~mask_mod.all(axis=0, keepdims=True)
            input_mod = np.where(mask_mod, np.nan, input_mod)

        if len(dates) != num_dates:
            input_mod = input_mod[slices[2]]
            input_mod = input_mod.reshape((num_dates, -1, *input_mod.shape[1:]))
            input_mod = np.nanmedian(input_mod, axis=1)
            dates_mod = dates[slices[2]]
            dates_mod = dates_mod.reshape((num_dates, -1, *dates_mod.shape[1:]))
            dates_mod = np.median(dates_mod, axis=1)

        if norm_fac is not None:
            input_mod /= norm_fac

        return input_mod, dates_mod

    def transform(
        self,
        inputs: dict[str, np.ndarray | list],
    ) -> dict[str, np.ndarray | list]:
        """Transform model inputs."""
        if self.transform is not None:
            inputs_transformed = {
                name_mod: inputs[name_mod].transpose([1, 2, 3, 0])
                for name_mod in self.filter_rasters
            }
            inputs_transformed = self.transform(
                **inputs_transformed,
                image=next(
                    iter(inputs_transformed.values()),
                ),  # add 'image' to make albumentations happy
            )
            inputs_transformed = {
                name_mod: inputs_transformed[name_mod].transpose([3, 0, 1, 2])
                for name_mod in self.filter_rasters
            }
            inputs.update(inputs_transformed)

        return inputs
