"""Base classes for modality and single dataset."""

from abc import ABC
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
class InputModConfig:
    """Input modality config."""

    resample_fac: float = 1.0
    num_dates: int = 1

    patch_size: PatchSizeConfig = MISSING
    norm_fac: float = MISSING
    num_bands: int = MISSING
    use_mask: bool = False


@dataclass
class TargetModConfig:
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
class DatasetConfig(ABC):
    """Generic dataset config."""

    target_patch_size: int | None = None

    def __post_init__(self) -> None:
        """Check validity of modalities and further define non configurable attrs."""
        for name_mod in self.filter_inputs + self.filter_targets:
            if name_mod not in self.__dict__:
                msg = f"Invalid modality name {name_mod}. Not an attribute."
                raise ValueError(msg)
        self.inputs = {
            name_mod: getattr(self, name_mod) for name_mod in self.filter_inputs
        }
        self.targets = {
            name_mod: getattr(self, name_mod) for name_mod in self.filter_targets
        }
