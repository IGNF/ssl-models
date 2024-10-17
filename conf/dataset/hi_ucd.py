"""Hi-UCD config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
)


@dataclass
class HiUCDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "Hi-UCD"
    weight: float = 1.0

    filter_input: list[str] = field(
        default_factory=lambda: ["image_1", "image_2"],
    )  # selected input modalities
    filter_target: list[str] = field(
        default_factory=lambda: ["change_detect"],
    )  # selected target modalities

    image_1: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=512,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            norm_fac=255.0,
        ),
    )
    image_2: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=512,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            norm_fac=255.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.gsd = 0.1

        self.change_detect = TargetModConfig(
            type_target="change_detect",
            num_classes=None,
            missing_val=None,
        )
        super().__post_init__()
