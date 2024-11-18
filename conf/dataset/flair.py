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
    rel_dir: str = "FLAIR_INC"
    weight: float = 1.0
    crop_meters: float = 250 * 0.2
    ref_input: str | None = "aerial"

    filter_inputs: list[str] = field(
        default_factory=lambda: ["aerial", "dtm_dsm", "spot", "s2", "s1"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["flair"],
    )  # selected target modalities

    aerial: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=4,
            norm_fac=255.0,
        ),
    )
    dtm_dsm: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            bands=2,
            norm_fac=1.0,
        ),
    )
    spot: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
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
    s1: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=2,
            num_dates=4,
            norm_fac=10000.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = FLAIRDataset
        self.ground_meters = 500 * 0.2

        self.flair = TargetRasterConfig(
            type_target="segment",
            num_classes=19,
            missing_val=-1,
        )
        super().__post_init__(
            resolutions_meters={
                "flair": 0.2,
                "aerial": 0.2,
                "dtm_dsm": 0.2,
                "spot": 1.0,
                "s2": 10.0,
                "s1": 10.0,
            },
        )
