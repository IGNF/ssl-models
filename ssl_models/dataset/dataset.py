"""Generic dataset module."""

import itertools
from abc import ABC

import albumentations as A  # noqa: N812
import numpy as np
import skimage
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

        self.inputs = dataset.inputs
        self.filter_inputs = dataset.filter_inputs
        self.filter_targets = dataset.filter_targets
        self.filter_rasters = dataset.filter_inputs + [
            name_mod
            for name_mod, mod in dataset.targets.items()
            if mod.type_target in RASTER_TARGETS
        ]
        self.image_size = dataset.image_size
        self.ground_size = dataset.ground_size

    def resample_and_normalize(
        self,
        input_mod: dict[str, np.ndarray],
        resample_fac: float = 1.0,
        num_bands: int | None = None,
        norm_fac: float | None = None,
    ) -> np.ndarray:
        if resample_fac != 1:
            if (
                not (input_mod.shape[3] / resample_fac).is_integer()
                or not (input_mod.shape[2] / resample_fac).is_integer()
            ):
                msg = f"Invalid resampling {resample_fac} for shape {input_mod.shape}."
                raise ValueError(msg)
            output_shape = (
                *input_mod.shape[:2],
                round(input_mod.shape[2] / resample_fac),
                round(input_mod.shape[3] / resample_fac),
            )
            input_mod = skimage.transform.resize(input_mod, output_shape, order=1)
        if num_bands != input_mod.shape[1]:
            if num_bands > input_mod.shape[1]:
                msg = (
                    f"Invalid number of bands {num_bands} for shape {input_mod.shape}."
                )
                raise ValueError(msg)
            input_mod = input_mod[:, :num_bands]
        if norm_fac is not None:
            input_mod /= norm_fac
        return input_mod

    def preprocess_input(
        self,
        inputs: dict[str, np.ndarray | list],
        name_mod: str,
        mod: DatasetConfig,
    ) -> dict[str, np.ndarray | list]:
        input_mod = inputs[name_mod].astype(np.float32)
        if mod.use_mask:  # mask
            mask_mod = inputs[f"{name_mod}_mask"].astype(bool)
            mask_mod = mask_mod.any(axis=-3, keepdims=True)  # either cloud or snow mask
            mask_mod = mask_mod & ~mask_mod.all(axis=0, keepdims=True)
            input_mod = np.where(mask_mod, np.nan, input_mod)

        dates_mod = inputs[f"{name_mod}_dates"]
        if len(dates_mod) < mod.num_dates:
            msg = (
                f"Invalid number of dates for {name_mod}."
                f"Cannot take {mod.num_dates} medians from {len(dates_mod)} dates."
            )
            raise ValueError(msg)

        chunks = np.array_split(np.argsort(dates_mod), mod.num_dates)
        inputs_mod = {}
        inputs_mod[name_mod] = np.stack(
            [
                np.nanmedian(
                    input_mod[chunk],
                    axis=0,
                )
                for chunk in chunks
            ],
        )
        inputs_mod[f"{name_mod}_dates"] = [
            [date_mod for idx, date_mod in enumerate(dates_mod) if idx in chunk]
            for chunk in chunks
        ]

        inputs_mod[name_mod] = self.resample_and_normalize(
            inputs_mod[name_mod],
            resample_fac=mod.resample_fac,
            num_bands=mod.num_bands,
            norm_fac=mod.norm_fac,
        )
        return inputs_mod

    def crop_raster(
        self,
        inputs: dict[str, np.ndarray | list],
    ) -> dict[str, np.ndarray | list]:
        inputs_mod = {}
        for axis in (-2, -1):
            sizes = {
                name_mod: inputs[name_mod].shape[axis]
                for name_mod in self.filter_rasters
            }
            for name_input, name_raster in itertools.product(
                self.filter_inputs,
                self.filter_rasters,
            ):
                size_input = sizes[name_input]
                size_raster = sizes[name_raster]
                if name_raster in self.filter_inputs:
                    if size_input % size_raster != 0 and size_raster % size_input != 0:
                        msg = "Sizes of inputs are not multiple of each other."
                        raise ValueError(msg)
                else:  # noqa: PLR5501
                    if size_raster % size_input != 0:
                        msg = "Size of target not multiple of size of input."
                        raise ValueError(msg)
            size_min, size_max = min(sizes.values()), max(sizes.values())
            image_size_min = (self.image_size * size_min) // size_max
            crop_min = self.rng.integers(size_min - image_size_min + 1)
            for name_mod, size in sizes.items():
                if not (size / size_min).is_integer():
                    msg = "Sizes of modalities are not multiple of each other."
                    raise ValueError(msg)
                crop = crop_min * (size // size_min)
                image_size = image_size_min * (size // size_min)
                inds = range(crop, crop + image_size)
                inputs_mod[name_mod] = np.take(inputs[name_mod], inds, axis=axis)
        return inputs_mod

    def preprocess(
        self,
        inputs: dict[str, np.ndarray | list],
    ) -> dict[str, np.ndarray | list]:
        """Preprocess model inputs."""
        inputs_processed = {}
        for name_mod, mod in self.inputs.items():
            inputs_mod = self.preprocess_input(
                inputs,
                name_mod,
                mod,
            )
            inputs_processed.update(inputs_mod)
        for name_mod in self.filter_targets:
            inputs_mod = {
                name_mod: inputs[name_mod],
                f"{name_mod}_dates": inputs[f"{name_mod}_dates"],
            }
            inputs_processed.update(inputs_mod)

        inputs_cropped = self.crop_raster(inputs_processed)
        inputs_processed.update(inputs_cropped)

        if self.transform is not None:
            inputs_transformed = {
                name_mod: inputs_processed[name_mod].transpose([1, 2, 3, 0])
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
        inputs_processed.update(inputs_transformed)

        return inputs_processed
