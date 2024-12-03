"""Mask conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class MaskConfig:  # noqa: D101
    mask_ratio: float = 0.75
    nb_mix_images: int = 4
    mask_mod: float | None = 0.5
    mask_bands: float | None = 0.5
    mask_dates: float | None = 0.5


model_store = store(group="mask")
model_store(MaskConfig, name="default_mask")
