"""S2-NAIP config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
)


@dataclass
class S2NAIPConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "s2-naip-urban"
    weight: float = 1.0

    filter_input: list[str] = field(
        default_factory=lambda: ["naip", "landsat", "s2", "s1"],
    )  # selected input modalities
    filter_target: list[str] = field(
        default_factory=lambda: ["osm"],
    )  # selected target modalities

    naip: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            norm_fac=255.0,
        ),
    )
    landsat: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=8, mixmae=2),
            norm_fac=10000.0,
            num_dates=4,
        ),
    )
    s2: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=8,
        ),
    )
    s1: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.gsd = 1.25

        self.osm = TargetModConfig(
            type_target="segment",
            num_classes=32,
            missing_val=None,
        )
        super().__post_init__()
