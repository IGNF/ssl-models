"""GeoPile data conf."""

from dataclasses import dataclass, field

from conf.datasets.utils import (
    InputModality,
    PatchSize,
    SingleDatasetConfig,
    TargetModality,
)


@dataclass
class GeoPileConfig(SingleDatasetConfig):  # noqa: D101
    rel_dir: str = "GeoPileV0"
    weight: float = 0.0

    input_modalities: list[str] = field(
        default_factory=lambda: ["rgb"],
    )  # selected input modalities
    target_modalities: list[str] = field(
        default_factory=lambda: ["mlrsnet", "patternnet", "rsd46_whu", "resisc45"],
    )  # selected target modalities

    rgb: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=256,
            patch_size=PatchSize(mae=16, mixmae=4),
            norm_fac=255.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable target modalities."""
        self.mlrsnet = TargetModality(
            type_target="classif",
            num_classes=46,
            missing_val=-1,
        )
        self.patternnet = TargetModality(
            type_target="classif",
            num_classes=38,
            missing_val=-1,
        )
        self.rsd46_whu = TargetModality(
            type_target="classif",
            num_classes=46,
            missing_val=-1,
        )
        self.resisc45 = TargetModality(
            type_target="classif",
            num_classes=45,
            missing_val=-1,
        )
        super().__post_init__()
