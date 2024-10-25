"""S2-NAIP config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.s2_naip import S2NAIPDataset


@dataclass
class S2NAIPConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "s2-naip-urban"
    weight: float = 1.0
    image_size: int = 256
    target_patch_size = 16

    filter_inputs: list[str] = field(
        default_factory=lambda: ["naip", "landsat", "s2", "s1"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["osm_seg"],
    )  # selected target modalities

    naip: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=4,
            norm_fac=255.0,
        ),
    )
    landsat: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=8, mixmae=2),
            bands=11,
            num_dates=4,
            norm_fac=10000.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=13,
            num_dates=8,
            norm_fac=10000.0,
        ),
    )
    s1: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=2,
            num_dates=4,
            norm_fac=10000.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = S2NAIPDataset

        self.osm_seg = TargetRasterConfig(
            type_target="segment",
            num_classes=32,
            missing_val=None,
        )

        self.ground_meters = 512 * 1.25
        self.crop_meters = 256 * 1.25

        super().__post_init__(
            resolutions_meters={
                "osm_seg": 1.25,
                "naip": 1.25,
                "landsat": 20.0,
                "s2": 10.0,
                "s1": 10.0,
            },
        )
