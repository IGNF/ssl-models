"""Augmentation conf.

Dataset-level augmentations (random crop and dates, transforms) are set in the
data config. This config holds model-specific, on-GPU augmentations.
"""

from dataclasses import dataclass, field

from hydra_zen import store


@store(group="augment", name="dino_augment")
@dataclass
class DinoAugmentConfig:
    """DINO multi-crop augmentation, see `ssl_models.dino.augment.MultiCropAugment`.

    `view_drop_mod` and `view_drop_dates` drop modalities and dates per view.
    """

    n_local_crops: int = 8
    global_crop_scale: list[float] = field(default_factory=lambda: [0.32, 1.0])
    local_crop_scale: list[float] = field(default_factory=lambda: [0.05, 0.32])
    local_size_ratio: float = 0.4375
    flip: bool = True
    jitter_prob: float = 0.8
    brightness: float = 0.2
    contrast: float = 0.2
    blur_min_size: int = 32
    view_drop_mod: float | None = 0.25
    view_drop_dates: float | None = 0.25
