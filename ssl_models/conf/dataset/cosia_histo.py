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
    remap_to_others_after_class: int | None = None
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
            num_classes=19,
            missing_val=-1,
            class_names=[
                "Bâtiment",
                "Serre",
                "Piscine",
                "Zone imperméable",
                "Zone perméable",
                "Sol nu",
                "Eau",
                "Neige",
                "Surface herbacée",
                "Culture",
                "Terre labourée",
                "Vigne",
                "Feuillu",
                "Conifère",
                "Broussaille",
                "Coupe",
                "Ligneux",
                "Mixte",
                "Autres",
            ],
            bands=1,
            seg_loss="cross_entropy",
        ),
    )

    def __post_init__(self) -> None:
        """Define non configurable attributes."""
        self.dataset_class = COSIAHistoDataset
        self.total_meters = 102.4

        # Optionally collapse classes (n+1..19) into the last class ("Autres").
        if self.remap_to_others_after_class is not None:
            max_kept_classes = self.cosia.num_classes - 1
            if not 1 <= self.remap_to_others_after_class <= max_kept_classes:
                msg = (
                    "remap_to_others_after_class must be in "
                    f"[1, {max_kept_classes}] when cosia.num_classes="
                    f"{self.cosia.num_classes}."
                )
                raise ValueError(msg)

            self.cosia.num_classes = self.remap_to_others_after_class + 1
            if self.cosia.class_names is not None:
                self.cosia.class_names = [
                    *self.cosia.class_names[: self.remap_to_others_after_class],
                    self.cosia.class_names[-1],
                ]

        # Keep band-selection user-facing as a flat list while using internal format.
        self.aerial.bands = [self.aerial_bands]

        super().__post_init__(
            resolutions_meters={
                "cosia": 0.2,
                "aerial": 0.2,
            },
        )

