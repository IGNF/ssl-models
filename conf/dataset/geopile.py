"""GeoPile config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
)
from ssl_models.dataset.geopile import GeoPileDataset


@dataclass
class GeoPileConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "GeoPileV0"
    weight: float = 0.0
    image_size: int = 256

    filter_inputs: list[str] = field(
        default_factory=lambda: ["rgb"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["mlrsnet", "patternnet", "rsd46_whu", "resisc45"],
    )  # selected target modalities

    rgb: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            norm_fac=255.0,
            num_bands=3,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = GeoPileDataset
        self.ground_size = None  # GSD set directly in loader as it's not fixed

        self.mlrsnet = TargetModConfig(
            type_target="classif",
            num_classes=46,
            missing_val=-1,
        )
        self.patternnet = TargetModConfig(
            type_target="classif",
            num_classes=38,
            missing_val=-1,
        )
        self.rsd46_whu = TargetModConfig(
            type_target="classif",
            num_classes=46,
            missing_val=-1,
        )
        self.resisc45 = TargetModConfig(
            type_target="classif",
            num_classes=45,
            missing_val=-1,
        )
        super().__post_init__()
