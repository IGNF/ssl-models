"""Hi-UCD config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
)
from ssl_models.dataset.hi_ucd import HiUCDDataset


@dataclass
class HiUCDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "Hi-UCD"
    weight: float = 1.0
    image_size: int = 512
    target_patch_size = 16

    filter_inputs: list[str] = field(
        default_factory=lambda: ["image_1", "image_2"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["change_detect"],
    )  # selected target modalities

    image_1: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            norm_fac=255.0,
            num_bands=3,
        ),
    )
    image_2: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            norm_fac=255.0,
            num_bands=3,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = HiUCDDataset
        self.ground_size = 512 * 0.1

        self.change_detect = TargetModConfig(
            type_target="change_detect",
            num_classes=None,
            missing_val=-1,
        )
        super().__post_init__()
