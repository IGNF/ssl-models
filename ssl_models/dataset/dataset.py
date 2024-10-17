"""Generic dataset module."""

from abc import ABC

import albumentations as A  # noqa: N812
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset

from conf.dataset.utils import (
    RASTER_TARGETS,
    DatasetConfig,
    InputModConfig,
    TargetModConfig,
)


class GenericDataset(Dataset, ABC):
    """Generic dataset."""

    def __init__(
        self,
        transform: A.Compose | None,
        dataset: DatasetConfig,
    ) -> None:
        self.rng = np.random.default_rng()

        self.gsd = dataset.gsd
        self.mods = {
            name_mod: getattr(dataset, name_mod)
            for name_mod in dataset.filter_input + dataset.filter_target
        }
        self.names_raster = [
            name_mod
            for name_mod, mod in self.mods.items()
            if isinstance(mod, InputModConfig)
            or isinstance(mod, TargetModConfig)
            and mod.type_target in RASTER_TARGETS
        ]
        self.transform = transform

    def resample_and_normalize(
        self,
        tensor_mod: np.ndarray,
        resample_fac: float = 1.0,
        norm_fac: float | None = None,
    ) -> np.ndarray:
        if resample_fac != 1:
            tensor_mod = ndimage.zoom(tensor_mod, zoom=1 / resample_fac, order=1)
        if norm_fac is not None:
            tensor_mod /= norm_fac
        return tensor_mod

    def preprocess_mod(
        self,
        inputs: dict[str, np.ndarray | list],
        name_mod: str,
        mod: InputModConfig,
    ) -> dict[str, np.ndarray | list]:
        H, W = inputs[name_mod] / mod.resample_fac  # noqa: N806
        if not H.is_integer() or not W.is_integer():
            msg = (
                f"Invalid resampling factor {mod.resample_fac} for modality {name_mod}."
                "Resulting height/width for resized arrays must be integers."
            )
            raise ValueError(msg)

        dates_mod = inputs[f"{name_mod}_dates"]

        input_mod = inputs[name_mod].astype(np.float32)
        input_mod = self.resample_and_normalize(
            input_mod,
            resample_fac=mod.resample_fac,
            norm_fac=mod.norm_fac,
        )
        if mod.use_mask:  # mask
            mask_mod = inputs[f"{name_mod}_mask"].astype(bool)
            mask_mod = self.resample_and_normalize(
                mask_mod,
                resample_fac=mod.resample_fac,
            )
            mask_mod = mask_mod & ~mask_mod.all(axis=0, keepdim=True)
            input_mod = np.where(mask_mod, np.nan, input_mod)

        inputs_mod = {}
        if len(dates_mod) == 1:
            inputs_mod[name_mod] = input_mod
            inputs_mod[f"{name_mod}_dates"] = dates_mod
        else:
            chunks = np.array_split(np.argsort(dates_mod), mod.num_dates)
            for idx_chunk, chunk in enumerate(chunks):  # apply medians in each chunk
                inputs_mod[f"{name_mod}_{idx_chunk}"] = np.nanmedian(
                    input_mod[chunk],
                    axis=0,
                )
                inputs_mod[f"{name_mod}_{idx_chunk}_dates"] = [
                    dates_mod[idx] for idx in chunk
                ]
        return inputs_mod

    def crop(
        self, inputs: dict[str, np.ndarray | list]
    ) -> dict[str, np.ndarray | list]:
        inputs_mod = {}
        for axis in (-2, -1):
            sizes = {
                name_mod: inputs[name_mod].shape[axis] for name_mod in self.names_raster
            }
            name_min = min(sizes, key=sizes.get)
            mod_min, size_min = self.mods[name_min], self.sizes[name_min]
            crop_min = self.rng.integers(size_min - mod_min.image_size + 1)
            for name_mod in self.names_raster:
                size_fac = sizes[name_mod] / size_min
                if not size_fac.is_integer():
                    msg = (
                        f"Size of modality {name_mod}"
                        f"is not a multiple of size of modality {name_min}"
                    )
                    raise ValueError(msg)
                crop = crop_min * round(size_fac)
                inds = range(crop, crop + self.mods[name_mod].image_size)
                inputs_mod[name_mod] = np.take(inputs[name_mod], inds, axis=axis)
        return inputs_mod

    def preprocess(
        self,
        inputs: dict[str, np.ndarray | list],
    ) -> dict[str, np.ndarray | list]:
        """Preprocess model inputs."""
        inputs_processed = {}
        for name_mod, mod in self.mods.items():
            inputs_mod = self.preprocess_mod(
                inputs,
                name_mod,
                mod,
            )
            inputs_processed.update(inputs_mod)

        inputs_cropped = self.crop(inputs_processed)
        inputs_processed.update(inputs_cropped)

        if self.transform is not None:
            inputs_transformed = {
                name_mod: inputs[name_mod].transpose([1, 2, 0])
                for name_mod in self.names_raster
            }
            inputs_transformed = self.transform(inputs_transformed)
            inputs_transformed = {
                name_mod: inputs[name_mod].transpose([2, 0, 1])
                for name_mod in self.names_raster
            }
        inputs_processed.update(inputs_transformed)

        return inputs
