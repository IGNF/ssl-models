"""Guyane config."""

from dataclasses import dataclass, field

from ssl_models.conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    register_dataset,
)
from ssl_models.dataset.guyane import GuyaneDataset


@register_dataset("guyane")
@dataclass
class GuyaneConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "guyane_pretrain"
    repeats: int = 1
    crop_meters: float = 1000
    grid_pos_enc: int | None = 160
    val_pretrain: bool = True
    test_pretrain: bool = True

    ref_input: str | None = "s2"  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: [
            "s2",
        ],
    )  # logged input modalities

    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "s2",
            "alos2",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: [],
    )  # selected target modalities

    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=100,
            patch_size=PatchSizeConfig(mae=5, mixmae=20),
            bands=10,
            norm_bands=[4, 4, 2],
            num_dates=4,
            mask_threshold=0.0,
            norm_fac=5000.0,
        ),
    )
    alos2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=40,
            patch_size=PatchSizeConfig(mae=2, mixmae=8),
            bands=2,
            norm_bands=[1, 1],
            num_dates=1,
            norm_fac=1000.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = GuyaneDataset
        self.total_meters = 1000

        super().__post_init__(
            resolutions_meters={
                "s2": 10,
                "alos2": 25,
            },
        )
