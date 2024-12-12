"""Hi-UCD config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.hi_ucd import HiUCDDataset


@dataclass
class HiUCDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "Hi-UCD"
    crop_meters: float = 512 * 0.1
    ref_input: str | None = "image_2"  # defines grid for raster targets

    filter_inputs: list[str] = field(
        default_factory=lambda: ["image_1", "image_2"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["hi_ucd"],
    )  # selected target modalities

    image_1: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=512,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            bands=3,
            norm_fac=255.0,
        ),
    )
    image_2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=512,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            bands=3,
            norm_fac=255.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = HiUCDDataset
        self.total_meters = 512 * 0.1

        self.hi_ucd = TargetRasterConfig(
            type_target="change_detect",
            num_classes=None,
            missing_val=-1,
            bands=[3],
        )
        super().__post_init__(
            resolutions_meters={"hi_ucd": 0.1, "image_1": 0.1, "image_2": 0.1},
        )
