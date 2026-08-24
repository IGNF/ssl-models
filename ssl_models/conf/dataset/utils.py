"""Base classes for modality and single dataset."""

from dataclasses import dataclass
from math import gcd

from hydra_zen import MISSING

ALLOWED_TARGETS = [
    "classif",
    "multilabel_classif",
    "segment",
    "change_detect",
]


@dataclass
class PatchSizeConfig:
    """Patch size config."""

    mae: int = MISSING
    mixmae: int = MISSING
    dinov2_imagenat: int = 14
    dinov2_sat: int = 16
    dofa: int = 16
    croma: int = 8


@dataclass
class RasterConfig:
    """Generic raster modality config."""

    bands: int | list[list[int]] = MISSING
    norm_bands: list[int] | None = None
    mask_threshold: float = 0.0
    num_dates: int = 1
    norm_fac: float | None = None
    log_scale: bool = False
    rescale_elev: bool = False
    name_embed: str | None = None
    class_names: list[str] | None = None


@dataclass
class InputConfig:
    """Generic input modality config."""

    image_size: int = MISSING
    patch_size: PatchSizeConfig = MISSING
    name_group: str | None = None


@dataclass
class TargetConfig:
    """Target modality config."""

    type_target: str = MISSING
    num_classes: int = MISSING
    missing_val: int = -1
    class_names: list[str] | None = None
    seg_loss: str = "cross_entropy"

    def __post_init__(self) -> None:
        """Check that attribute values are valid."""
        if self.type_target not in ALLOWED_TARGETS:
            msg = (
                f"Invalid target {self.type_target}."
                f"Expected one of {ALLOWED_TARGETS}"
            )
            raise ValueError(msg)
        allowed_seg_losses = {"cross_entropy", "hungarian"}
        if self.seg_loss not in allowed_seg_losses:
            msg = (
                f"Invalid seg_loss {self.seg_loss}. "
                f"Expected one of {sorted(allowed_seg_losses)}"
            )
            raise ValueError(msg)


@dataclass
class InputRasterConfig(RasterConfig, InputConfig):
    """Input raster modality config."""


@dataclass
class TargetRasterConfig(RasterConfig, TargetConfig):
    """Target raster modality config."""


@dataclass
class DatasetConfig:
    """Generic dataset config."""

    def _set_resolutions(self, resolutions_meters: dict[str, int]) -> None:
        self.sizes = {}
        for name_mod, resolution_meters in resolutions_meters.items():
            if name_mod not in self.__dict__:
                msg = f"Invalid modality {name_mod} specified in resolution."
                raise ValueError(msg)

            mod = getattr(self, name_mod)
            mod.resolution_meters = float(resolution_meters)

            size = self.total_meters / mod.resolution_meters
            is_filtered = name_mod in self.filter_inputs + self.filter_targets
            if not size.is_integer() and is_filtered:
                msg = f"Modality {name_mod}'s resolution does not divide image extent."
                raise ValueError(msg)

            self.sizes[name_mod] = round(size)

        size_gcd = gcd(*self.sizes.values())
        crop_gcd = self.crop_meters / self.total_meters * size_gcd
        if not crop_gcd.is_integer():
            msg = (
                f"Crop meters does not correspond to an integer number of pixels."
                f"Use a multiple of {self.total_meters / size_gcd}."
            )
            raise ValueError(msg)

        crop_gcd_int = round(crop_gcd)
        if crop_gcd_int <= 0:
            msg = (
                f"Invalid crop_meters={self.crop_meters}: crop size in gcd pixels "
                f"must be strictly positive (got {crop_gcd_int})."
            )
            raise ValueError(msg)
        if crop_gcd_int > size_gcd:
            msg = (
                f"Invalid crop_meters={self.crop_meters}: crop window exceeds image "
                f"extent (crop_gcd={crop_gcd_int}, size_gcd={size_gcd}). "
                f"Set crop_meters <= total_meters ({self.total_meters})."
            )
            raise ValueError(msg)

        self.size_gcd = size_gcd
        self.crop_gcd = crop_gcd_int

    def _set_groups(self) -> list[tuple]:
        groups = []
        for name_mod, mod in self.inputs.items():
            name_group = mod.name_group if mod.name_group is not None else name_mod
            groups.append((name_mod, name_group))

        return groups

    def __post_init__(self, resolutions_meters: dict[str, int]) -> None:
        """Check validity of modalities and further define non configurable attrs."""
        self._set_resolutions(resolutions_meters)

        self.log_inputs = [
            log_input
            for log_input in self.log_inputs
            if log_input in self.filter_inputs
        ]
        if not self.log_inputs:
            self.log_inputs = self.filter_inputs

        if self.ref_input and self.ref_input not in self.filter_inputs:
            msg = f"Ref input {self.ref_input} is not selected."
            raise ValueError(msg)

        self.inputs = {}
        for name_mod in self.filter_inputs:
            if name_mod not in self.__dict__:
                msg = f"Invalid modality name {name_mod}. Not an attribute."
                raise ValueError(msg)
            self.inputs[name_mod] = getattr(self, name_mod)

        self.targets = {}
        for name_mod in self.filter_targets:
            if name_mod not in self.__dict__:
                msg = f"Invalid modality name {name_mod}. Not an attribute."
                raise ValueError(msg)
            self.targets[name_mod] = getattr(self, name_mod)

        self.rasters = {
            name_mod: mod
            for name_mod, mod in (*self.inputs.items(), *self.targets.items())
            if isinstance(mod, RasterConfig)
        }

        self.groups = self._set_groups()

