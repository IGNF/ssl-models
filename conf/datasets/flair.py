"""GeoPile data conf."""

from dataclasses import dataclass, field

from conf.datasets.utils import (
    InputModality,
    PatchSize,
    SingleDatasetConfig,
    TargetModality,
)


@dataclass
class FLAIRConfig(SingleDatasetConfig):  # noqa: D101
    rel_dir: str = "FLAIR_INC"
    weight: float = 1.0

    input_modalities: list[str] = field(
        default_factory=lambda: ["aerial", "dtm_dsm", "spot", "s2", "s1"],
    )  # selected input modalities
    target_modalities: list[str] = field(
        default_factory=lambda: ["segment"],
    )  # selected target modalities

    aerial: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=256,
            patch_size=PatchSize(mae=16, mixmae=4),
            norm_fac=255.0,
        ),
    )
    dtm_dsm: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=256,
            patch_size=PatchSize(mae=32, mixmae=8),
            norm_fac=None,
        ),
    )
    spot: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=32,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=255.0,
        ),
    )
    s2: InputModality = field(
        default_factory=lambda: InputModality(
            pixel_size_fac=0.32,  # convert to 3.2m resolution
            image_size=16,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
        ),
    )
    s1: InputModality = field(
        default_factory=lambda: InputModality(
            pixel_size_fac=0.32,  # convert to 3.2m resolution
            image_size=16,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable target modality."""
        self.segment = TargetModality(
            type_target="segment",
            num_classes=19,
            missing_val=-1,
        )
        super().__post_init__()
