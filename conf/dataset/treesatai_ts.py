"""TreeSatAI-TS config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputModConfig,
    PatchSizeConfig,
    TargetModConfig,
)
from ssl_models.dataset.treesatai_ts import TreeSatAITSDataset


@dataclass
class TreeSatAITSConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "TreeSatAI-TS"
    weight: float = 1.0
    image_size: int = 256

    filter_inputs: list[str] = field(
        default_factory=lambda: ["aerial", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["multilabel_classif"],
    )  # selected target modalities

    aerial: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=300 / 256.0,  # resample to 256 x 256 pixels
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            norm_fac=255.0,
            num_bands=4,
        ),
    )
    s2: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=6 / 16.0,  # resample to 16 x 16 pixels
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=4,
            use_mask=True,
            num_bands=10,
        ),
    )
    s1_asc: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=6 / 16.0,  # resample to 16 x 16 pixels
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
            num_bands=2,
        ),
    )
    s1_des: InputModConfig = field(
        default_factory=lambda: InputModConfig(
            resample_fac=0.32,  # resample to 3.2m resolution
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            norm_fac=10000.0,
            num_dates=2,
            num_bands=2,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = TreeSatAITSDataset
        self.ground_size = 300 * 0.2

        self.multilabel_classif = TargetModConfig(
            type_target="multilabel_classif",
            num_classes=15,
            missing_val=-1,
        )
        super().__post_init__()
