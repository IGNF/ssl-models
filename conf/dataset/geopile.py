"""GeoPile config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetConfig,
)
from ssl_models.dataset.geopile import GeoPileDataset


@dataclass
class GeoPileConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "GeoPileV0"
    crop_meters: float = 256.0
    ref_input: str | None = None

    filter_inputs: list[str] = field(
        default_factory=lambda: ["rgb"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["mlrsnet", "patternnet", "rsd46_whu", "resisc45"],
    )  # selected target modalities

    rgb: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=3,
            norm_fac=255.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = GeoPileDataset
        self.total_meters = 256.0

        self.mlrsnet = TargetConfig(
            type_target="classif",
            num_classes=46,
            missing_val=-1,
        )
        self.patternnet = TargetConfig(
            type_target="classif",
            num_classes=38,
            missing_val=-1,
        )
        self.rsd46_whu = TargetConfig(
            type_target="classif",
            num_classes=46,
            missing_val=-1,
        )
        self.resisc45 = TargetConfig(
            type_target="classif",
            num_classes=45,
            missing_val=-1,
        )
        super().__post_init__(resolutions_meters={"rgb": 1.0})
