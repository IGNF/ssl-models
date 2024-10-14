"""Base classes for modality and single dataset."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hydra_zen import MISSING

ALLOWED_TARGETS = [
    "classif",
    "multilabel_classif",
    "segment",
    "change_detect",
]


@dataclass
class PatchSize:
    """Patch size data class."""

    mae: int = MISSING
    mixmae: int = MISSING


@dataclass
class InputModality:
    """Input modality data class."""

    pixel_size_fac: float = 1.0
    num_dates: int = 1

    image_size: int = MISSING
    patch_size: PatchSize = MISSING
    norm_fac: float = MISSING

    def __post_init__(self) -> None:
        """Check that attribute values are valid."""
        if (
            self.image_size % self.patch_size.mae
            or self.image_size % self.patch_size.mixmae
        ):
            msg = "Invalid choice of patch size. It must be divisor of the image size."
            raise ValueError(msg)


@dataclass
class TargetModality:
    """Target modality data class."""

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
class SingleDatasetConfig(ABC):
    """Generic dataset config."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    def __post_init__(self) -> None:
        """Check that input/target names are valid."""
        for name in self.input_modalities + self.target_modalities:
            if name not in self.__dict__:
                msg = f"Invalid input/target name {name}. Not an attribute."
                raise ValueError(msg)
