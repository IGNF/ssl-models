"""Base classes for modality and single dataset."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hydra_zen import MISSING

ALLOWED_TARGETS = [
    "classif",
    "multilabel_classif",
    "regress",
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

    image_size: int = MISSING
    patch_size: PatchSizeConfig = MISSING
    norm_fac: float = MISSING
    use_mask: bool = False

    def __post_init__(self) -> None:
        """Check that attribute values are valid."""
        if (
            self.image_size % self.patch_size.mae
            or self.image_size % self.patch_size.mixmae
        ):
            msg = "Invalid choice of patch size. It must be divisor of the image size."
            raise ValueError(msg)


@dataclass
class TargetModConfig:
    """Target modality config."""

    type_target: str = MISSING
    num_classes: int | None = None
    missing_val: int | None = None

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

    @abstractmethod
    def __init__(self) -> None:
        pass

    def __post_init__(self) -> None:
        """Check that input/target names are valid."""
        for name_mod in self.filter_input + self.filter_target:
            if name_mod not in self.__dict__:
                msg = f"Invalid modality name {name_mod}. Not an attribute."
                raise ValueError(msg)
