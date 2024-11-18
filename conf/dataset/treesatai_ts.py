"""TreeSatAI-TS config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetConfig,
)
from ssl_models.dataset.treesatai_ts import TreeSatAITSDataset


@dataclass
class TreeSatAITSConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "TreeSatAI-TS"
    weight: float = 1.0
    crop_meters: float = 60.0
    ref_input: str | None = None

    filter_inputs: list[str] = field(
        default_factory=lambda: ["aerial", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["treesat_mlc"],
    )  # selected target modalities

    aerial: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=4,
            norm_fac=255.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=10,
            num_dates=4,
            mask_threshold=60.0,
            norm_fac=5000.0,
        ),
    )
    s1_asc: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=2,
            num_dates=1,
            norm_fac=0.25,
        ),
    )
    s1_des: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=2,
            num_dates=1,
            norm_fac=0.25,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = TreeSatAITSDataset
        self.ground_meters = 60.0

        self.treesat_mlc = TargetConfig(
            type_target="multilabel_classif",
            num_classes=15,
            missing_val=-1,
        )
        super().__post_init__(
            resolutions_meters={
                "aerial": 0.2,
                "s2": 10.0,
                "s1_asc": 10.0,
                "s1_des": 10.0,
            },
        )
