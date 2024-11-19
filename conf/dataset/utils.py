"""Base classes for modality and single dataset."""

from dataclasses import dataclass

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


@dataclass
class RasterConfig:
    """Generic raster modality config."""

    image_size: int = MISSING
    patch_size: PatchSizeConfig = MISSING
    bands: list[int] | int = MISSING


@dataclass
class InputConfig:
    """Generic input modality config."""

    use_mask: bool = False
    num_dates: int = 1
    norm_fac: float = MISSING


@dataclass
class TargetConfig:
    """Target modality config."""

    type_target: str = MISSING
    num_classes: int = MISSING
    missing_val: int = -1

    def __post_init__(self) -> None:
        """Check that attribute values are valid."""
        if self.type_target not in ALLOWED_TARGETS:
            msg = (
                f"Invalid target {self.type_target}."
                f"Expected one of {ALLOWED_TARGETS}"
            )
            raise ValueError(msg)


@dataclass
class InputRasterConfig(RasterConfig, InputConfig):
    """Input raster modality config."""


@dataclass
class TargetRasterConfig(RasterConfig, TargetConfig):
    """Target raster modality config."""

    num_dates: int = 1
    colormap: list[list[int]] = MISSING


@dataclass
class DatasetConfig:
    """Generic dataset config."""

    def __post_init__(self, resolutions_meters: dict[str, int]) -> None:
        """Check validity of modalities and further define non configurable attrs."""
        for name_mod, resolution_meters in resolutions_meters.items():
            if name_mod not in self.__dict__:
                msg = f"Invalid modality {name_mod} specified in resolution."
                raise ValueError(msg)

            ground_pixels = self.ground_meters / float(resolution_meters)
            crop_pixels = self.crop_meters / float(resolution_meters)
            is_filtered = name_mod in self.filter_inputs + self.filter_targets
            if not ground_pixels.is_integer() and is_filtered:
                msg = f"Modality {name_mod}'s resolution does not divide image extent."
                raise ValueError(msg)

            if not crop_pixels.is_integer() and is_filtered:
                msg = f"Modality {name_mod}'s resolution does not divide crop extent."
                raise ValueError(msg)

            mod = getattr(self, name_mod)
            mod.resolution_meters = float(resolution_meters)

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
