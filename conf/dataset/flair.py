"""FLAIR config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.flair import FLAIRDataset


@dataclass
class FLAIRConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "FLAIR-INC_TOY"
    crop_meters: float = 256 * 0.2
    ref_input: str | None = "aerial"  # defines grid for raster targets

    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "aerial",
            "dem",
            "spot",
            "s2",
            "s1_asc",
            "s1_des",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: [
            "flair_cosia",
            # "flair_lpis"
        ],
    )  # selected target modalities

    aerial: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=4,
            norm_fac=255.0,
        ),
    )
    dem: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=8),
            bands=2,
            norm_fac=1.0,
        ),
    )
    spot: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=4,
            norm_fac=255.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=10,
            num_dates=4,
            mask_threshold=0.0,
            norm_fac=10000.0,
        ),
    )
    s1_asc: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=2,
            num_dates=4,
            norm_fac=20.0,
        ),
    )
    s1_des: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=2,
            num_dates=4,
            norm_fac=20.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = FLAIRDataset
        self.total_meters = 102.4

        self.flair_cosia = TargetRasterConfig(
            type_target="segment",
            num_classes=19,
            missing_val=-1,
            bands=1,
        )

        self.flair_lpis = TargetRasterConfig(
            type_target="segment",
            num_classes=45,
            missing_val=-1,
            bands=1,
        )

        super().__post_init__(
            resolutions_meters={
                "flair_cosia": 0.2,
                "flair_lpis": 0.2,
                "aerial": 0.2,
                "dem": 0.2,
                "spot": 1.6,
                "s2": 10.24,
                "s1_asc": 10.24,
                "s1_des": 10.24,
            },
        )
