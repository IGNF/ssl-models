"""Flair 2 config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.flair_2 import FLAIR2Dataset


@dataclass
class FLAIR2Config(DatasetConfig):  # noqa: D101
    rel_dir: str = "FLAIR_2"
    balance_pretrain: bool = False
    crop_meters: float = 256 * 0.2
    ref_input: str | None = "aerial"  # defines grid for raster targets

    log_inputs: list[str] = field(
        default_factory=lambda: ["aerial"],
    )  # logged input modalities
    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "aerial",
            "s2",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: [
            "flair_2_cosia",
        ],
    )  # selected target modalities

    aerial: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            bands=4,
            norm_fac=1000.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=5,
            patch_size=PatchSizeConfig(mae=1, mixmae=1),
            bands=10,
            num_dates=16,
            mask_threshold=0.0,
            norm_fac=5000.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = FLAIR2Dataset
        self.total_meters = 102.4

        self.flair_2_cosia = TargetRasterConfig(
            type_target="segment",
            num_classes=19,
            missing_val=-1,
            bands=1,
        )

        super().__post_init__(
            resolutions_meters={
                "flair_2_cosia": 0.2,
                "aerial": 0.2,
                "s2": 10.24,
            },
        )
