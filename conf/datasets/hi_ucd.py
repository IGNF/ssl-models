"""GeoPile data conf."""

from dataclasses import dataclass, field

from conf.datasets.utils import (
    InputModality,
    PatchSize,
    SingleDatasetConfig,
    TargetModality,
)


@dataclass
class HiUCDConfig(SingleDatasetConfig):  # noqa: D101
    rel_dir: str = "Hi-UCD"
    weight: float = 1.0

    input_modalities: list[str] = field(
        default_factory=lambda: ["image_1", "image_2"],
    )  # selected input modalities
    target_modalities: list[str] = field(
        default_factory=lambda: ["change_detect"],
    )  # selected target modalities

    image_1: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=512,
            patch_size=PatchSize(mae=32, mixmae=8),
            norm_fac=255.0,
        ),
    )
    image_2: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=512,
            patch_size=PatchSize(mae=32, mixmae=8),
            norm_fac=255.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable target modality."""
        self.change_detect = TargetModality(
            type_target="change_detect",
            num_classes=None,
            missing_val=None,
        )
        super().__post_init__()
