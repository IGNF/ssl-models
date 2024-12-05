"""PASTIS-HD config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.pastis_hd import PASTISHDDataset


@dataclass
class PASTISHDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "PASTIS-HD"
    weight: float = 1.0
    crop_meters: float = 160 * 1.0
    ref_input: str | None = "s2"

    filter_inputs: list[str] = field(
        default_factory=lambda: ["spot", "s2"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["pastis_seg", "pastis_mlc"],
    )  # selected target modalities

    spot: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=160,
            patch_size=PatchSizeConfig(mae=8, mixmae=4),
            bands=3,
            norm_fac=255.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=10,
            num_dates=8,
            patch_dates=4,
            norm_fac=10000.0,
        ),
    )
    s1_asc: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
            num_dates=2,
            patch_dates=4,
            norm_fac=20.0,
        ),
    )
    s1_des: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
            num_dates=2,
            patch_dates=4,
            norm_fac=20.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = PASTISHDDataset
        self.ground_meters = 1280 * 1.0

        self.pastis_seg = TargetRasterConfig(
            type_target="segment",
            num_classes=19,
            missing_val=19,
            bands=1,
        )
        self.pastis_mlc = TargetConfig(
            type_target="multilabel_classif",
            num_classes=18,
        )
        super().__post_init__(
            resolutions_meters={
                "pastis_seg": 10,
                "spot": 1.0,
                "s2": 10.0,
                "s1_asc": 10.0,
                "s1_des": 10.0,
            },
        )
