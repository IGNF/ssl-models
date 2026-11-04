"""PASTIS-HD config."""

from dataclasses import dataclass, field

from ssl_models.conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
    register_dataset,
)
from ssl_models.dataset.pastis_hd import PASTISHDDataset


@register_dataset("pastis_hd")
@dataclass
class PASTISHDConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "PASTIS-HD"
    csv_dir: str | None = None
    balance_pretrain: bool = False
    val_pretrain: bool = True
    filter_percent: int | None = None
    fold: int | None = None
    repeats: int = 8
    crop_meters: float = 160
    grid_pos_enc: int | None = 256

    ref_input: str | None = "s2"  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: ["spot"],
    )  # logged input modalities
    filter_inputs: list[str] = field(
        default_factory=lambda: ["spot", "s2", "s1_asc", "s1_des"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["pastis_seg"],
    )  # selected target modalities

    spot: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=160,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=3,
            norm_fac=255.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=10,
            norm_bands=[4, 4, 2],
            num_dates=16,
            norm_fac=10000.0,
        ),
    )
    s1_asc: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=[[0, 1]],
            norm_bands=[1, 1],
            num_dates=4,
            norm_fac=20.0,
            name_group="sen1",
        ),
    )
    s1_des: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=[[0, 1]],
            norm_bands=[1, 1],
            num_dates=4,
            norm_fac=20.0,
            name_group="sen1",
        ),
    )

    pastis_seg: TargetRasterConfig = field(
        default_factory=lambda: TargetRasterConfig(
            type_target="segment",
            num_classes=19,
            missing_val=19,
            bands=1,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = PASTISHDDataset
        self.total_meters = 1280 * 1.0

        super().__post_init__(
            resolutions_meters={
                "pastis_seg": 10,
                "spot": 1.0,
                "s2": 10.0,
                "s1_asc": 10.0,
                "s1_des": 10.0,
            },
        )
