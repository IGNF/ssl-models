"""Circa config."""

from dataclasses import dataclass, field

from conf.dataset.utils import DatasetConfig, InputRasterConfig, PatchSizeConfig
from ssl_models.dataset.circa import CircaDataset


@dataclass
class CircaConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "Circa"
    dataset_local: bool = False
    csv_dir: str | None = None
    version: str | None = None
    val_pretrain: bool = False
    filter_percent: int | None = None
    repeats: int = 68  # 68
    crop_meters: float = 163.84 * 2

    ref_input: str | None = None  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: ["s2_zoom"],
    )  # logged input modalities

    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "s2_global",
            "s1_asc_global",
            "s1_des_global",
            "s2_zoom",
            "s1_asc_zoom",
            "s1_des_zoom",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=list,
    )  # selected target modalities

    s1_asc_global: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(vit=4, swin=1),
            bands=4,
            num_dates=4,
            norm_fac=1.0,
            name_group="all",
        ),
    )
    s1_des_global: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(vit=4, swin=1),
            bands=4,
            num_dates=4,
            norm_fac=1.0,
            name_group="all",
        ),
    )
    s2_global: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(vit=4, swin=1),
            bands=[[2, 1, 0, 3, 4, 5, 6, 7, 8, 9]],
            num_dates=4,
            norm_bands=[4, 4, 2],
            mask_bands=[10, 11],
            mask_threshold=0.0,
            norm_fac=5000.0,
            offset=-1000,
            name_group="all",
        ),
    )
    s1_asc_zoom: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(vit=4, swin=1),
            bands=4,
            num_dates=4,
            norm_fac=1.0,
            name_group="all",
        ),
    )
    s1_des_zoom: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(vit=4, swin=1),
            bands=4,
            num_dates=4,
            norm_fac=1.0,
            name_group="all",
        ),
    )
    s2_zoom: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=32,
            patch_size=PatchSizeConfig(vit=4, swin=1),
            bands=[[2, 1, 0, 3, 4, 5, 6, 7, 8, 9]],
            num_dates=4,
            norm_bands=[4, 4, 2],
            ndvi_bands=[2, 6],
            mask_bands=[10, 11],
            fake_cloud_val=150,
            mask_threshold=0.0,
            norm_fac=5000.0,
            offset=-1000,
            name_group="all",
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
                "s2_zoom": 10.24,
                "s1_asc_zoom": 10.24,
                "s1_des_zoom": 10.24,
            },
        )
