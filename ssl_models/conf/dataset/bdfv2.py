"""BDFV2 config."""

from dataclasses import dataclass, field

from ssl_models.conf.dataset.utils import (DatasetConfig, InputRasterConfig,
                                           PatchSizeConfig, TargetConfig)
from ssl_models.dataset.bdfv2 import BDFV2Dataset


@dataclass
class BDFV2Config(DatasetConfig):  # noqa: D101
    rel_dir: str = "BDFV2"
    csv_dir: str | None = None
    val_pretrain: bool = True
    test_pretrain: bool = True
    repeats: int = 1
    crop_meters: float = 50
    grid_pos_enc: int | None = 80

    ref_input: str | None = None
    log_inputs: list[str] = field(
        default_factory=lambda: [
            "aerial",
        ],
    )  # logged input modalities

    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "aerial",
            "dem",
            "s2",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["bdfv2"],
    )  # selected target modalities

    aerial: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=250,
            patch_size=PatchSizeConfig(mae=10, mixmae=2),
            bands=[[0, 1, 2, 3]],
            norm_bands=[3, 1],
            norm_fac=255.0,
        ),
    )
    dem: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=250,
            patch_size=PatchSizeConfig(mae=25, mixmae=5),
            bands=2,
            norm_fac=1000.0,
            rescale_elev=True,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=10,
            patch_size=PatchSizeConfig(mae=2, mixmae=1),
            bands=10,
            norm_bands=[4, 4, 2],
            num_dates=12,
            norm_fac=5000.0,
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = BDFV2Dataset
        self.total_meters = 50

        self.bdfv2 = TargetConfig(
            type_target="classif",
            num_classes=30,
            missing_val=-1,
        )

        super().__post_init__(
            resolutions_meters={
                "aerial": 0.2,
                "dem": 0.2,
                "s2": 10.0,
            },
        )
