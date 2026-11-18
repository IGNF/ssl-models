"""COSIA-Histo config."""

from dataclasses import dataclass, field

from ssl_models.conf.dataset.utils import (
    DatasetConfig,
    InputRasterConfig,
    PatchSizeConfig,
    TargetRasterConfig,
    register_dataset,
)
from ssl_models.dataset.cosia_histo import COSIAHistoDataset

CLASS_NAMES = [
    "building",
    "greenhouse",
    "swimming pool",
    "impervious surface",
    "pervious surface",
    "bare soil",
    "water",
    "snow",
    "herbaceous vegetation",
    "agricultural land",
    "plowed land",
    "vineyard",
    "deciduous",
    "coniferous",
    "brushwood",
    "clear-cut",
    "woody",
    "mixed",
]

NUM_CLASSES = len(CLASS_NAMES)

# The raw label also has an "unannotated" class, mapped to `MISSING_VAL`.
MISSING_VAL = -1

# Coarser nomenclatures: (class names, new class id of each raw COSIA label).
NOMENCLATURES = {
    "cosia_artif_large": (
        [
            "artificialise_bati",
            "artificialise_sol",
            "naturel_sol_nu",
            "naturel_eau",
            "naturel_sol_vegetation",
            "agricole",
            "ligneux",
        ],
        [0, 0, 1, 1, 1, 2, 3, 2, 4, 5, 5, 5, 6, 6, 4, 6, 6, 6, MISSING_VAL],
    ),
    "cosia_artif_medium": (
        [
            "artificialise_bati",
            "artificialise_sol",
            "naturel_sol",
            "naturel_eau",
            "agricole",
            "ligneux",
        ],
        [0, 0, 1, 1, 1, 2, 3, 2, 2, 4, 4, 4, 5, 5, 2, 5, 5, 5, MISSING_VAL],
    ),
    "cosia_artif_small": (
        [
            "artificialise",
            "naturel",
            "agricole",
            "ligneux",
        ],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 3, 3, 3, MISSING_VAL],
    ),
}


def nomenclature_target(name: str) -> TargetRasterConfig:
    """Return the target config of a coarser nomenclature."""
    class_names, _ = NOMENCLATURES[name]
    return TargetRasterConfig(
        type_target="segment",
        num_classes=len(class_names),
        missing_val=MISSING_VAL,
        bands=1,
        class_names=list(class_names),
    )


@register_dataset("cosia_histo")
@dataclass
class COSIAHistoConfig(DatasetConfig):  # noqa: D101
    rel_dir: str = "COSIA-HISTO"
    gpkg_dir: str | None = None
    modalities_gpkg: str = "GLOBAL_ALL_MTD_MODALITIES.gpkg"
    split_gpkg: str = "GLOBAL_ALL_MTD_SPLIT.gpkg"
    fid_col: str = "fid"
    split_col: str = "split"
    aerial_path_col: str = "aerial_rgbi"
    target_path_col: str = "ground_truth"
    repeats: int = 1
    crop_meters: float = 102.4
    grid_pos_enc: int | None = 160
    default_date: str = "20000101"
    default_date_fmt: str = "%Y%m%d"
    # Merges the `cosia` classes from this id onwards into a single "other" class.
    remap_to_others_after_class: int | None = None
    aerial_bands: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    ref_input: str | None = "aerial"  # defines grid for raster targets
    log_inputs: list[str] = field(
        default_factory=lambda: ["aerial"],
    )  # logged input modalities
    filter_inputs: list[str] = field(
        default_factory=lambda: ["aerial"],
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
            num_classes=NUM_CLASSES,
            missing_val=MISSING_VAL,
            bands=1,
            class_names=list(CLASS_NAMES),
        ),
    )
    cosia_artif_large: TargetRasterConfig = field(
        default_factory=lambda: nomenclature_target("cosia_artif_large"),
    )
    cosia_artif_medium: TargetRasterConfig = field(
        default_factory=lambda: nomenclature_target("cosia_artif_medium"),
    )
    cosia_artif_small: TargetRasterConfig = field(
        default_factory=lambda: nomenclature_target("cosia_artif_small"),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = COSIAHistoDataset
        self.total_meters = 102.4

        # New class id of each raw COSIA label, for each target.
        self.class_remaps = {
            name: list(remap) for name, (_, remap) in NOMENCLATURES.items()
        }

        n_kept = self.remap_to_others_after_class
        if n_kept is not None and not 1 <= n_kept < NUM_CLASSES:
            msg = (
                f"remap_to_others_after_class must be in [1, {NUM_CLASSES - 1}], "
                f"got {n_kept}."
            )
            raise ValueError(msg)
        self.class_remaps["cosia"] = [
            *(min(c, n_kept or NUM_CLASSES) for c in range(NUM_CLASSES)),
            MISSING_VAL,
        ]
        if n_kept is not None:
            self.cosia.num_classes = n_kept + 1
            self.cosia.class_names = [*CLASS_NAMES[:n_kept], "other"]
            self.cosia.check_class_names()

        # Keep band-selection user-facing as a flat list while using internal format.
        self.aerial.bands = [self.aerial_bands]

        super().__post_init__(
            resolutions_meters={
                "aerial": 0.2,
                "cosia": 0.2,
                "cosia_artif_large": 0.2,
                "cosia_artif_medium": 0.2,
                "cosia_artif_small": 0.2,
            },
        )
