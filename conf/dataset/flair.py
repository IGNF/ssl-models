"""FLAIR config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
)


@dataclass
class FLAIRConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "FLAIR_INC"
    weight: float = 1.0

    filter_input: list[str] = field(
        default_factory=lambda: ["aerial", "dtm_dsm", "spot", "s2", "s1"],
    )  # selected input modalities
    filter_target: list[str] = field(
        default_factory=lambda: ["segment"],
    )  # selected target modalities

    aerial: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            norm_fac=255.0,
        ),
    )
    dtm_dsm: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            norm_fac=None,
        ),
    )
    spot: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=255.0,
        ),
    )
    s2: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.32,  # resample to 3.2m resolution
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
            use_mask=True,
        ),
    )
    s1: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.32,  # resample to 3.2m resolution
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.gsd = 0.2
        self.segment = TargetModConfig(
            type_target="segment",
            num_classes=19,
            missing_val=-1,
        )
        super().__post_init__()
