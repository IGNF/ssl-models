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
    class_remap: dict[int, int] | None = field(
        default_factory=lambda: {
        1: 0,   # "Bâtiment" → "Background"
        2: 0,   # "Zone perméable" → "Background"
        3: 0,   # "Zone imperméable" → "Background"
        4: 0,   # "Piscine" → "Background"
        5: 0,   # "Sol nu" → "Background"
        6: 0,   # "Eau" → "Background"
        9: 0,   # "Coupe" → "Background"
        7: 0,   # "Neige" → "Background"
        11: 0, # "Mixte" → "Background"
        12: 0, # "Ligneux" → "Background"
        17: 0, # "Terre labourée" → "Background"
        18: 0, # "Serre" → "Background"
        }
    )

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
            norm_fac=5000.0,
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
                "Background",
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

        if self.class_remap is not None:
            original_num_classes = self.tempetes.num_classes
            for src, dst in self.class_remap.items():
                if not (0 <= src < original_num_classes):
                    msg = (
                        f"class_remap key {src} is out of range "
                        f"[0, {original_num_classes})."
                    )
                    raise ValueError(msg)
                if dst < 0:
                    msg = f"class_remap value {dst} must be >= 0."
                    raise ValueError(msg)

            # Build identity LUT then apply overrides.
            lut = list(range(original_num_classes))
            for src, dst in self.class_remap.items():
                lut[src] = dst

            new_num_classes = max(lut) + 1

            if self.tempetes.class_names is not None:
                new_class_names: list[str | None] = [None] * new_num_classes
                for i, name in enumerate(self.tempetes.class_names):
                    out_i = lut[i]
                    if new_class_names[out_i] is None:
                        new_class_names[out_i] = name
                self.tempetes.class_names = [
                    n if n is not None else "" for n in new_class_names
                ]

            self._original_num_classes = original_num_classes
            self.tempetes.num_classes = new_num_classes

        super().__post_init__(
            resolutions_meters={
                "tempetes": 0.5,
                "pleiades": 0.5,
                "s2": 8, # 8m resolution for 256 px patch, 10.24 for 512 px patch.
            },
        )
