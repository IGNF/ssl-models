"""FLAIR config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
)
from ssl_models.dataset.flair import FLAIRDataset


@dataclass
class FLAIRConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "FLAIR_INC"
    weight: float = 1.0
    image_size: int = 256

    filter_inputs: list[str] = field(
        default_factory=lambda: ["aerial", "dtm_dsm", "spot", "s2", "s1"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["segment"],
    )  # selected target modalities

    aerial: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            norm_fac=255.0,
            num_bands=4,
        ),
    )
    dtm_dsm: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            norm_fac=None,
            num_bands=2,
        ),
    )
    spot: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=255.0,
            num_bands=3,
        ),
    )
    s2: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.32,  # resample to 3.2m resolution
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
            use_mask=True,
            num_bands=10,
        ),
    )
    s1: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.32,  # resample to 3.2m resolution
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
            num_bands=2,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = FLAIRDataset
        self.ground_size = 512 * 0.2

        self.segment = TargetModConfig(
            type_target="segment",
            num_classes=19,
            missing_val=-1,
        )
        super().__post_init__()
