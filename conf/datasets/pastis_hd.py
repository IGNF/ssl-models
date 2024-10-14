"""GeoPile data conf."""

from dataclasses import dataclass, field

from conf.datasets.utils import (
    InputModality,
    PatchSize,
    SingleDatasetConfig,
    TargetModality,
)


@dataclass
class PASTISHDConfig(SingleDatasetConfig):  # noqa: D101
    rel_dir: str = "PASTIS-HD"
    weight: float = 1.0

    input_modalities: list[str] = field(
        default_factory=lambda: ["spot", "s2", "s1"],
    )  # selected input modalities
    target_modalities: list[str] = field(
        default_factory=lambda: ["segment"],
    )  # selected target modalities

    spot: InputModality = field(
        default_factory=lambda: InputModality(
            image_size=256,
            patch_size=PatchSize(mae=16, mixmae=4),
            norm_fac=255.0,
        ),
    )
    s2: InputModality = field(
        default_factory=lambda: InputModality(
            pixel_size_fac=0.8,  # convert to 8m resolution
            image_size=32,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
        ),
    )
    s1_asc: InputModality = field(
        default_factory=lambda: InputModality(
            pixel_size_fac_fac=0.8,  # convert to 8m resolution
            image_size=32,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
        ),
    )
    s1_des: InputModality = field(
        default_factory=lambda: InputModality(
            pixel_size_fac=0.8,  # convert to 8m resolution
            image_size=32,
            patch_size=PatchSize(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable target modality."""
        self.segment = TargetModality(
            type_target="segment",
            num_classes=19,
            missing_val=19,
        )
        super().__post_init__()
