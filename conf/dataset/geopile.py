"""GeoPile config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (DatasetConfig, InputModConfig, PatchSizeConfig,
                                TargetModConfig)


@dataclass
class GeoPileConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "GeoPileV0"
    weight: float = 0.0

    filter_input: list[str] = field(
        default_factory=lambda: ["rgb"],
    )  # selected input modalities
    filter_target: list[str] = field(
        default_factory=lambda: ["mlrsnet", "patternnet", "rsd46_whu", "resisc45"],
    )  # selected target modalities

    rgb: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            norm_fac=255.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable target modalities."""
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
