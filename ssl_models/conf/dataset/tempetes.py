"""Tempetes config."""

from dataclasses import dataclass, field

from ssl_models.conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
    register_dataset,
)
from ssl_models.dataset.tempetes import TempetesDataset

# Number of classes before remapping.
NUM_CLASSES = 22

# Maps original class ids to new ones; unlisted classes keep their id.
DEFAULT_CLASS_REMAP = {
    1: 0,  # Bâtiment
    2: 0,  # Zone perméable
    3: 0,  # Zone imperméable
    4: 0,  # Piscine
    5: 0,  # Sol nu
    6: 0,  # Eau
    7: 0,  # Neige
    8: 1,  # Conifère
    9: 0,  # Coupe
    10: 2,  # Feuillus
    11: 0,  # Mixte
    12: 0,  # Ligneux
    13: 3,  # Broussailles
    14: 4,  # Vigne
    15: 5,  # Pelouse
    16: 6,  # Culture
    17: 0,  # Terre labourée
    18: 0,  # Serre
    19: 0,  # Inconnu
    20: 7,  # Feuillus nus
    21: 8,  # Sinistre
}

CLASS_NAMES = [
    "background",
    "building",
    "pervious surface",
    "impervious surface",
    "swimming pool",
    "bare soil",
    "water",
    "snow",
    "coniferous",
    "clear-cut",
    "deciduous",
    "mixed",
    "woody",
    "brushwood",
    "vineyard",
    "grassland",
    "cropland",
    "plowed land",
    "greenhouse",
    "unknown",
    "bare deciduous",
    "damaged",
]

# Names of the classes obtained with `DEFAULT_CLASS_REMAP`.
REMAPPED_CLASS_NAMES = [
    "other",
    "coniferous",
    "deciduous",
    "brushwood",
    "vineyard",
    "grassland",
    "cropland",
    "bare deciduous",
    "damaged",
]


@register_dataset("tempetes")
@dataclass
class TempetesConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "Tempetes"
    csv_dir: str | None = None
    fold: int | None = None
    balance_pretrain: bool = False
    val_pretrain: bool = True
    test_pretrain: bool = False
    filter_percent: int | None = None
    repeats: int = 1
    crop_meters: float = 128
    grid_pos_enc: int | None = 160
    class_remap: dict[int, int] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_REMAP),
    )  # applied to the `tempetes_remap` target

    ref_input: str | None = "pleiades"  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: ["pleiades"],
    )  # logged input modalities
    filter_inputs: list[str] = field(
        default_factory=lambda: ["pleiades", "s2"],
    )  # selected input modalities
    filter_targets: list[str] = field(
        default_factory=lambda: ["tempetes_remap"],
    )  # selected target modalities

    pleiades: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=256,
            patch_size=PatchSizeConfig(mae=16, mixmae=4),
            bands=[[0, 1, 2, 3]],
            norm_bands=[1, 3],
            norm_fac=5000.0,
        ),
    )
    s2: InputRasterConfig = field(
        default_factory=lambda: InputRasterConfig(
            image_size=16,  # 16 px for a 256 px pleiades patch, 25 px for 512 px
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
            num_classes=NUM_CLASSES,
            missing_val=-1,
            bands=1,
            class_names=list(CLASS_NAMES),
        ),
    )
    tempetes_remap: TargetRasterConfig = field(
        default_factory=lambda: TargetRasterConfig(
            type_target="segment",
            num_classes=len(REMAPPED_CLASS_NAMES),
            missing_val=-1,
            bands=1,
            class_names=list(REMAPPED_CLASS_NAMES),
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = TempetesDataset
        self.total_meters = 128

        for src, dst in self.class_remap.items():
            if not 0 <= src < NUM_CLASSES:
                msg = f"class_remap key {src} is out of range [0, {NUM_CLASSES})."
                raise ValueError(msg)
            if dst < 0:
                msg = f"class_remap value {dst} must be >= 0."
                raise ValueError(msg)

        lut = list(range(NUM_CLASSES))
        for src, dst in self.class_remap.items():
            lut[src] = dst
        self.tempetes_remap.num_classes = max(lut) + 1
        # Set `tempetes_remap.class_names` along with a custom `class_remap`.
        self.tempetes_remap.check_class_names()

        super().__post_init__(
            resolutions_meters={
                "tempetes": 0.5,
                "tempetes_remap": 0.5,
                "pleiades": 0.5,
                "s2": 8,  # 8 m for a 256 px patch, 10.24 m for 512 px
            },
        )
