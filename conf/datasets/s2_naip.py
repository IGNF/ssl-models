"""GeoPile data conf."""

from dataclasses import dataclass, field

from conf.datasets.utils import (
    InputModality,
    PatchSize,
    SingleDatasetConfig,
    TargetModality,
)


@dataclass
class S2NAIPConfig(SingleDatasetConfig):  # noqa: D101
    rel_dir: str = "s2-naip-urban"
    weight: float = 1.0

    input_modalities: list[str] = field(
        default_factory=lambda: ["naip", "landsat", "s2", "s1"],
    )  # selected input modalities
    target_modalities: list[str] = field(
        default_factory=lambda: ["openstreetmap"],
    )  # selected target modalities

    naip: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=256,
            patch_size=PatchSize(mae=16, mixmae=4),
            norm_fac=255.0,
        ),
    )
    landsat: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=32,
            patch_size=PatchSize(mae=8, mixmae=2),
            norm_fac=10000.0,
            num_dates=4,
        ),
    )
    s2: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=32,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=8,
        ),
    )
    s1: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=32,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable target modality."""
        self.openstreetmap = TargetModality(
            type_target="segment",
            num_classes=32,
            missing_val=None,
        )
        super().__post_init__()
