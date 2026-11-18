"""Augmentation conf used by the Lightning modules only.

Augmentations are applied on GPU by `ssl_models.train.augment.Augment`, which is
instantiated with `Augment(dataset, **vars(augment))`: parameters left out of a
config take the defaults of `Augment` (no crop, no jitter, no blur, a single
global view and no local view). By default, a single view of the full extent,
with flip (also at val/test). DINO uses several global and local views.
Read-time augmentations (random crop and dates) are set in the data config.
"""

from dataclasses import dataclass, field


@dataclass
class AugmentConfig:
    """Single-view GPU augmentation, see `ssl_models.train.augment.Augment`."""

    # geometric (shared by all modalities and dates of a view)
    interpolate: str = "bilinear"
    flip: bool = True


@dataclass
class DINOv3AugmentConfig:
    """DINOv3 multi-crop augmentation, see `ssl_models.train.augment.Augment`.

    `*_scale` is the area fraction range of the random crop boxes of the views of
    a type. `*_blur_probs` gives the blur probability of each view of a type, the
    last value being used for the remaining views.
    """

    # geometric (shared by all modalities and dates of a view)
    interpolate: str = "bilinear"
    flip: bool = True
    ratio: list[float] = field(default_factory=lambda: [3 / 4, 4 / 3])
    # photometric (after rescaling, inputs only)
    jitter_prob: float = 0.8
    brightness: float = 0.2
    contrast: float = 0.2
    blur_sigma: list[float] = field(default_factory=lambda: [0.1, 2.0])
    blur_min_size: int = 32
    # global views
    n_global: int = 2
    global_scale: list[float] = field(default_factory=lambda: [0.32, 1.0])
    global_blur_probs: list[float] = field(default_factory=lambda: [1.0, 0.1])
    # local views
    n_local: int = 8
    local_scale: list[float] = field(default_factory=lambda: [0.05, 0.32])
    local_blur_probs: list[float] = field(default_factory=lambda: [0.5])
    local_size_ratio: float = 0.4375  # local view size / image size
