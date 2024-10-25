"""Base classes for modality and single dataset."""

from dataclasses import dataclass

from hydra_zen import MISSING

ALLOWED_TARGETS = [
    "classif",
    "multilabel_classif",
    "segment",
    "change_detect",
]
RASTER_TARGETS = [
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


@dataclass
class DatasetConfig:
    """Generic dataset config."""

    def __post_init__(self, resolutions_meters: dict[str, int]) -> None:
        """Check validity of modalities and further define non configurable attrs."""
        for name_mod, resolution_meters in resolutions_meters.items():
            if name_mod not in self.__dict__:
                msg = f"Invalid modality name {name_mod}. Not an attribute."
                raise ValueError(msg)
            getattr(self, name_mod).resolution_meters = float(resolution_meters)

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
