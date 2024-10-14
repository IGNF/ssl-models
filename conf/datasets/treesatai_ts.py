"""GeoPile data conf."""

from dataclasses import dataclass, field

from conf.datasets.utils import (
    InputModality,
    PatchSize,
    SingleDatasetConfig,
    TargetModality,
)


@dataclass
class TreeSatAITSConfig(SingleDatasetConfig):  # noqa: D101
    rel_dir: str = "TreeSatAI-TS"
    weight: float = 1.0

    input_modalities: list[str] = field(
        default_factory=lambda: ["aerial", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    target_modalities: list[str] = field(
        default_factory=lambda: ["multilabel_classif"],
    )  # selected target modalities

    aerial: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=256,
            patch_size=PatchSize(mae=16, mixmae=4),
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
    s1_asc: InputModality = field(
        default_factory=lambda: InputModality(
            pixel_size_fac=0.32,  # convert to 3.2m resolution
            image_size=16,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
        ),
    )
    s1_des: InputModality = field(
        default_factory=lambda: InputModality(
            pixel_size_fac=0.32,  # convert to 3.2m resolution
            image_size=16,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable target modality."""
        self.multilabel_classif = TargetModality(
            type_target="multilabel_classif",
            num_classes=20,
            missing_val=None,
        )
        super().__post_init__()
