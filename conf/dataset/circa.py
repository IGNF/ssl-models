"""Circa config."""

from dataclasses import dataclass, field

from conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
)
from ssl_models.dataset.circa import CircaDataset


@dataclass
class CircaConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "Circa"
    csv_dir: str | None = None
    version: str | None = None
    balance_pretrain: bool = False
    val_pretrain: bool = False
    filter_percent: int | None = None
    repeats: int = 137
    crop_meters: float = 163.84

    ref_input: str | None = "s1_asc_zoom"  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: ["s2_zoom"],
    )  # logged input modalities

    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "s2_global",
            "s2_cloud_mask_global",
            "s1_asc_global",
            "s1_des_global",
            "s2_zoom",
            "s2_cloud_mask_zoom",
            "s1_asc_zoom",
            "s1_des_zoom",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=list,
    )  # selected target modalities

    s1_asc_global: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=4,
            num_dates=4,
            norm_fac=1.0,
            name_group="sen1_global",
        ),
    )
    s1_des_global: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=4,
            num_dates=4,
            norm_fac=1.0,
            name_group="sen1_global",
        ),
    )
    s2_global: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=10,
            norm_bands=[4, 6],
            num_dates=4,
            mask_threshold=0.0,
            norm_fac=5000.0,
        ),
    )
    s2_cloud_mask_global: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=[[11]],
            num_dates=4,
            mask_threshold=0.0,
        ),
    )
    s1_asc_zoom: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=4,
            num_dates=4,
            norm_fac=1.0,
            name_group="sen1_zoom",
        ),
    )
    s1_des_zoom: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=4,
            num_dates=4,
            norm_fac=1.0,
            name_group="sen1_zoom",
        ),
    )
    s2_zoom: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=10,
            num_dates=4,
            norm_bands=[4, 6],
            norm_fac=5000.0,
        ),
    )
    s2_cloud_mask_zoom: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=[[11]],
            num_dates=4,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = CircaDataset
        self.total_meters = 22446.08

        super().__post_init__(
            resolutions_meters={
                "s2_global": 10.24,
                "s1_asc_global": 10.24,
                "s1_des_global": 10.24,
                "s2_cloud_mask_global": 10.24,
                "s2_zoom": 10.24,
                "s1_asc_zoom": 10.24,
                "s1_des_zoom": 10.24,
                "s2_cloud_mask_zoom": 10.24,
            },
        )
