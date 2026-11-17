"""Mask conf."""

from dataclasses import dataclass, field


@dataclass
class MiMMaskConfig:
    """MiM masking, see `ssl_models.train.mim.MiMModule`."""

    mask_ratio: float = 0.75
    nb_mix_images: int = 4
    mask_scale: float = 0.0
    mask_mod: float | None = 0.25
    mask_bands: float | None = None
    mask_dates: float | None = 0.25
    mask_loc: float | None = 0.25


@dataclass
class DINOv3MaskConfig:
    """iBOT masking, see `ssl_models.train.dino.DINOModule`."""

    ibot_mask_prob: float = 0.5
    ibot_mask_ratio: list[float] = field(default_factory=lambda: [0.1, 0.5])
    ibot_blob_grid: int = 4
    ibot_mask_shared: float = 1.0
