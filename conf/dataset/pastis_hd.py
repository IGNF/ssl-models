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
    balance_pretrain: bool = False
    crop_meters: float = 320
    ref_input: str | None = "s2"  # defines grid for raster targets

    filter_inputs: list[str] = field(
        default_factory=lambda: ["spot", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["pastis_seg", "pastis_mlc"],
    )  # selected target modalities

    spot: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=320,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            bands=3,
            norm_fac=255.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=10,
            num_dates=16,
            norm_fac=10000.0,
        ),
    )
    s1_asc: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
            num_dates=4,
            norm_fac=20.0,
            name_group="sen1",
        ),
    )
    s1_des: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
            num_dates=4,
            norm_fac=20.0,
            name_group="sen1",
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = PASTISHDDataset
        self.total_meters = 1280 * 1.0

        self.pastis_seg = TargetRasterConfig(
            type_target="segment",
            num_classes=19,
            missing_val=19,
            bands=1,
            colormap=[
                (0, 0, 0),
                (174, 199, 232),
                (255, 127, 14),
                (255, 187, 120),
                (44, 160, 44),
                (152, 223, 138),
                (214, 39, 40),
                (255, 152, 150),
                (148, 103, 189),
                (197, 176, 213),
                (140, 86, 75),
                (196, 156, 148),
                (227, 119, 194),
                (247, 182, 210),
                (127, 127, 127),
                (199, 199, 199),
                (188, 189, 34),
                (219, 219, 141),
                (23, 190, 207),
                (255, 255, 255),
            ],
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
