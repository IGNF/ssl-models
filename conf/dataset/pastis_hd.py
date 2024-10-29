"""PASTIS-HD config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.pastis_hd import PASTISHDDataset


@dataclass
class PASTISHDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "PASTIS-HD"
    weight: float = 1.0
    crop_meters: float = 250 * 1.0

    filter_inputs: list[str] = field(
        default_factory=lambda: ["spot", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["pastis_seg"],
    )  # selected target modalities

    spot: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=3,
            norm_fac=255.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=10,
            num_dates=4,
            norm_fac=10000.0,
        ),
    )
    s1_asc: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
            num_dates=2,
            norm_fac=10000.0,
        ),
    )
    s1_des: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
            num_dates=2,
            norm_fac=10000.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = PASTISHDDataset
        self.ground_meters = 1280 * 1.0

        self.pastis_seg = TargetRasterConfig(
            type_target="segment",
            num_classes=19,
            missing_val=19,
        )
        super().__post_init__(
            resolutions_meters={
                "pastis_seg": 1.0,
                "spot": 1.0,
                "s2": 10.0,
                "s1_asc": 10.0,
                "s1_des": 10.0,
            },
        )
