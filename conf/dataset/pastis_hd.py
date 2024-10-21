"""PASTIS-HD config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
)
from ssl_models.dataset.pastis_hd import PASTISHDDataset


@dataclass
class PASTISHDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "PASTIS-HD"
    weight: float = 1.0
    image_size: int = 256
    target_patch_size = 16

    filter_inputs: list[str] = field(
        default_factory=lambda: ["spot", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["segment"],
    )  # selected target modalities

    spot: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            norm_fac=255.0,
            num_bands=3,
        ),
    )
    s2: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.8,  # resample to 8m resolution
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
            num_bands=10,
        ),
    )
    s1_asc: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.8,  # resample to 8m resolution
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
            num_bands=3,
        ),
    )
    s1_des: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.8,  # resample to 8m resolution
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
            num_bands=3,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = PASTISHDDataset
        self.gsd = 1280 * 1

        self.segment = TargetModConfig(
            type_target="segment",
            num_classes=19,
            missing_val=19,
        )
        super().__post_init__()
