"""PASTIS-HD config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (DatasetConfig, InputModConfig, PatchSizeConfig,
                                TargetModConfig)


@dataclass
class PASTISHDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "PASTIS-HD"
    weight: float = 1.0

    filter_input: list[str] = field(
        default_factory=lambda: ["spot", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_target: list[str] = field(
        default_factory=lambda: ["segment"],
    )  # selected target modalities

    spot: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            norm_fac=255.0,
        ),
    )
    s2: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.8,  # resample to 8m resolution
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
        ),
    )
    s1_asc: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.8,  # resample to 8m resolution
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
        ),
    )
    s1_des: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.8,  # resample to 8m resolution
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable target modality."""
        self.segment = TargetModConfig(
            type_target="segment",
            num_classes=19,
            missing_val=19,
        )
        super().__post_init__()
