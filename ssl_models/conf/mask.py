"""Mask conf."""

from dataclasses import dataclass, field


@dataclass
class MAEMaskConfig:
    """MAE masking, see `ssl_models.ssl.mae.MAE`."""

    mask_ratio: float = 0.75
    mask_scale: float = 0.0
    mask_mod: float | None = 0.25
    mask_bands: float | None = None
    mask_dates: float | None = 0.25
    mask_loc: float | None = 0.25


@dataclass
class DINOv3MaskConfig:
    """iBOT masking and per-view drop of modalities / dates.

    See `ssl_models.ssl.dino.DINOv3`.
    """

    ibot_mask_prob: float = 0.5
    ibot_mask_ratio: list[float] = field(default_factory=lambda: [0.1, 0.5])
    ibot_blob_grid: int = 4
    ibot_mask_shared: float = 1.0
    view_drop_mod: float | None = 0.25
    view_drop_dates: float | None = 0.25
