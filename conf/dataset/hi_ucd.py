"""Hi-UCD config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.hi_ucd import HiUCDDataset


@dataclass
class HiUCDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "Hi-UCD"
    crop_meters: float = 512 * 0.1
    ref_input: str | None = "image_2"  # defines grid for raster targets

    filter_inputs: list[str] = field(
        default_factory=lambda: ["image_1", "image_2"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: [
            "hi_ucd_change_detect",
            "hi_ucd_seg_1",
            "hi_ucd_seg_2",
        ],
    )  # selected target modalities

    image_1: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=512,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            bands=3,
            norm_fac=255.0,
        ),
    )
    image_2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=512,
            patch_size=PatchSizeConfig(mae=32, mixmae=8),
            bands=3,
            norm_fac=255.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = HiUCDDataset
        self.balance_pretrain = False
        self.total_meters = 512 * 0.1

        self.hi_ucd_change_detect = TargetRasterConfig(
            type_target="change_detect",
            num_classes=1,
            missing_val=-1,
            bands=[[-1]],
            colormap=[(255, 255, 255), (0, 0, 0), (220, 0, 0)],
        )

        seg_colormap = [
            (255, 255, 255),
            (0, 153, 255),
            (202, 255, 122),
            (230, 0, 0),
            (230, 0, 255),
            (255, 230, 0),
            (255, 181, 197),
            (0, 255, 230),
            (175, 122, 255),
            (26, 255, 0),
        ]
        self.hi_ucd_seg_1 = TargetRasterConfig(
            type_target="segment",
            num_classes=9,
            missing_val=-1,
            bands=[[0]],
            colormap=seg_colormap,
        )
        self.hi_ucd_seg_2 = TargetRasterConfig(
            type_target="segment",
            num_classes=9,
            missing_val=-1,
            bands=[[1]],
            colormap=seg_colormap,
        )
        super().__post_init__(
            resolutions_meters={
                "hi_ucd_change_detect": 0.1,
                "hi_ucd_seg_1": 0.1,
                "hi_ucd_seg_2": 0.1,
                "image_1": 0.1,
                "image_2": 0.1,
            },
        )
