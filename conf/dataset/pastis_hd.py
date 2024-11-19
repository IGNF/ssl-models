"""PASTIS-HD config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.pastis_hd import PASTISHDDataset


@dataclass
class PASTISHDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "PASTIS-HD"
    weight: float = 1.0
    crop_meters: float = 320 * 1.0
    ref_input: str | None = "spot"

    filter_inputs: list[str] = field(
        default_factory=lambda: ["spot", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["pastis_seg"],
    )  # selected target modalities

    spot: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=3,
            norm_fac=255.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=10,
            num_dates=4,
            norm_fac=10000.0,
        ),
    )
    s1_asc: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
            num_dates=2,
            norm_fac=20.0,
        ),
    )
    s1_des: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(mae=4, mixmae=1),
            bands=3,
            num_dates=2,
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
            bands=[0],
            num_dates=1,
            colormap=[
                [0, 0, 0],
                [0.6823529411764706, 0.7803921568627451, 0.9098039215686274],
                [1.0, 0.4980392156862745, 0.054901960784313725],
                [1.0, 0.7333333333333333, 0.47058823529411764],
                [0.17254901960784313, 0.6274509803921569, 0.17254901960784313],
                [0.596078431372549, 0.8745098039215686, 0.5411764705882353],
                [0.8392156862745098, 0.15294117647058825, 0.1568627450980392],
                [1.0, 0.596078431372549, 0.5882352941176471],
                [0.5803921568627451, 0.403921568627451, 0.7411764705882353],
                [0.7725490196078432, 0.6901960784313725, 0.8352941176470589],
                [0.5490196078431373, 0.33725490196078434, 0.29411764705882354],
                [0.7686274509803922, 0.611764705882353, 0.5803921568627451],
                [0.8901960784313725, 0.4666666666666667, 0.7607843137254902],
                [0.9686274509803922, 0.7137254901960784, 0.8235294117647058],
                [0.4980392156862745, 0.4980392156862745, 0.4980392156862745],
                [0.7803921568627451, 0.7803921568627451, 0.7803921568627451],
                [0.7372549019607844, 0.7411764705882353, 0.13333333333333333],
                [0.8588235294117647, 0.8588235294117647, 0.5529411764705883],
                [0.09019607843137255, 0.7450980392156863, 0.8117647058823529],
                [1, 1, 1],
            ],
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
