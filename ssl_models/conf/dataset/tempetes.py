"""Tempetes config."""

from dataclasses import dataclass, field

from ssl_models.conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
)
from ssl_models.dataset.tempetes import TempetesDataset


@dataclass
class TempetesConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "Tempetes"
    json_dir: str | None = None
    version: str | None = None
    s2_l3a: bool = False
    balance_pretrain: bool = False
    val_pretrain: bool = True
    test_pretrain: bool = False
    filter_percent: int | None = None
    repeats: int = 1
    crop_meters: float = 128
    grid_pos_enc: int | None = 160

    ref_input: str | None = "pleiades"  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: [
            "pleiades",
        ],
    )  # logged input modalities

    filter_inputs: list[str] = field(
        default_factory=lambda: [
            "pleiades",
            "s2",
        ],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["tempetes"],
    )  # selected target modalities

    pleiades: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=[[0, 1, 2, 3]],
            norm_bands=[1, 3],
            norm_fac=10000.0,
        ),
    )

    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16, # s2=16px for 256 px pleiades patch, s2=25px for 512 px pleiades patch.
            patch_size=PatchSizeConfig(mae=4, mixmae=4),
            bands=10,
            norm_bands=[4, 4, 2],
            num_dates=16,
            mask_threshold=0.0,
            norm_fac=5000.0,
        ),
    )

    tempetes: TargetRasterConfig = field(
        default_factory=lambda: TargetRasterConfig(
            type_target="segment",
            num_classes=22,
            missing_val=-1,
            bands=1,
            class_names=[
                "Non défini",
                "Bâtiment",
                "Zone perméable",
                "Zone imperméable",
                "Piscine",
                "Sol nu",
                "Eau",
                "Neige",
                "Conifères",
                "Coupe",
                "Feuillus",
                "Mixte",
                "Ligneux",
                "Broussaille",
                "Vigne",
                "Pelouse",
                "Culture",
                "Terre labourée",
                "Serre",
                "Inconnu",
                "Feuillus nus",
                "Sinistre",
            ],

        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = TempetesDataset
        self.total_meters = 128

        super().__post_init__(
            resolutions_meters={
                "tempetes": 0.5,
                "pleiades": 0.5,
                "s2": 8, # 8m resolution for 256 px patch, 10.24 for 512 px patch.
            },
        )