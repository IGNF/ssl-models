"""TreeSatAI-TS config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
)


@dataclass
class TreeSatAITSConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "TreeSatAI-TS"
    weight: float = 1.0

    filter_input: list[str] = field(
        default_factory=lambda: ["aerial", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_target: list[str] = field(
        default_factory=lambda: ["multilabel_classif", "regress"],
    )  # selected target modalities

    aerial: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
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
    s1_asc: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.32,  # resample to 3.2m resolution
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
        ),
    )
    s1_des: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.32,  # resample to 3.2m resolution
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.gsd = 0.2

        self.multilabel_classif = TargetModConfig(
            type_target="multilabel_classif",
            num_classes=20,
            missing_val=None,
        )
        self.regress = TargetModConfig(
            type_target="regress",
            num_classes=20,
            missing_val=None,
        )
        super().__post_init__()
