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
    crop_meters: float = 256 * 1.25
    ref_input: str | None = "naip"  # defines grid for raster targets

    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "naip",
            "landsat_10m",
            "landsat_20m",
            "s2_10m",
            "s2_20m",
            "s2_40m",
            "s1",
        ],
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
    landsat_10m: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=1,
            num_dates=4,
            norm_fac=10000.0,
        ),
    )
    landsat_20m: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=10,
            num_dates=4,
            norm_fac=10000.0,
        ),
    )
    s2_10m: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=4,
            num_dates=8,
            norm_fac=10000.0,
        ),
    )
    s2_20m: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=6,
            num_dates=8,
            norm_fac=10000.0,
        ),
    )
    s2_40m: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=8,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
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
            norm_fac=20.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = S2NAIPDataset
        self.total_meters = 512 * 1.25

        self.osm_seg = TargetRasterConfig(
            type_target="segment",
            num_classes=6,
            missing_val=-1,
        )
        super().__post_init__(
            resolutions_meters={
                "osm_seg": 1.25,
                "naip": 1.25,
                "landsat_10m": 10.0,
                "landsat_20m": 20.0,
                "s2_10m": 10.0,
                "s2_20m": 20.0,
                "s2_40m": 40.0,
                "s1": 10.0,
            },
        )
