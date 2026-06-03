"""COSIA-Histo config."""

from dataclasses import dataclass, field

from ssl_models.conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.cosia_histo import COSIAHistoDataset


@dataclass
class COSIAHistoConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "COSIA-HISTO"
    gpkg_dir: str | None = None
    modalities_gpkg: str = "GLOBAL_ALL_MTD_MODALITIES"
    split_gpkg: str = "GLOBAL_ALL_MTD_SPLIT"
    fid_col: str = "fid"
    split_col: str = "split"
    aerial_path_col: str = "aerial_rgbi"
    target_path_col: str = "ground_truth"
    repeats: int = 1
    crop_meters: float = 102.4
    grid_pos_enc: int | None = 160
    default_date: str = "20000101"
    default_date_fmt: str = "%Y%m%d"
    aerial_bands: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    ref_input: str | None = "aerial"  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: [
            "aerial",
        ],
    )  # logged input modalities
    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "aerial",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["cosia"],
    )  # selected target modalities

    aerial: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=512,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=[[0, 1, 2, 3]],
            norm_bands=[1, 3],
            norm_fac=255.0,
        ),
    )

    cosia: TargetRasterConfig = field(
        default_factory=lambda: TargetRasterConfig(
            type_target="segment",
            num_classes=15,
            missing_val=-1,
            bands=1,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = COSIAHistoDataset
        self.total_meters = 102.4

        # Keep band-selection user-facing as a flat list while using internal format.
        self.aerial.bands = [self.aerial_bands]

        super().__post_init__(
            resolutions_meters={
                "cosia": 0.2,
                "aerial": 0.2,
            },
        )