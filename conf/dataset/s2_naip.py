"""S2-NAIP config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
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
        default_factory=lambda: ["osm"],
    )  # selected target modalities

    naip: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            norm_fac=255.0,
            num_bands=4,
        ),
    )
    landsat: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=8, mixmae=2),
            norm_fac=10000.0,
            num_dates=4,
            num_bands=11,
        ),
    )
    s2: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=8,
            num_bands=13,
        ),
    )
    s1: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
            num_bands=2,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = S2NAIPDataset
        self.ground_size = 512 * 1.25

        self.osm = TargetModConfig(
            type_target="segment",
            num_classes=32,
            missing_val=None,
        )
        super().__post_init__()
