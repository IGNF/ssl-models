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

    bands: int | list[list[int]] = MISSING
    mask_threshold: float = 100.0
    num_dates: int = 1
    norm_fac: float | None = None


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

    def _set_resolutions(self, resolutions_meters: dict[str, int]) -> None:
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

    def _set_groups(self) -> list[tuple]:
        name_groups = set()
        groups = []
        for name_mod, mod in self.inputs.items():
            name_group = mod.name_group if mod.name_group is not None else name_mod
            name_groups.add(name_group)
            groups.append((name_mod, name_group))

        for name_group in name_groups:
            mods = (
                mod
                for name_mod, mod in self.inputs.items()
                if (name_mod, name_group) in groups
            )
            if len({mod.num_dates for mod in mods}) > 1:
                msg = "Grouped modalities must have the same number of dates"
                raise ValueError(msg)
            if len({mod.image_size for mod in mods}) > 1:
                msg = "Grouped modalities must have the same image sizes"
                raise ValueError(msg)
            if (
                len({mod.patch_size.mae for mod in mods}) > 1
                or len({mod.patch_size.mixmae for mod in mods}) > 1
            ):
                msg = "Grouped modalities must have the same patch sizes"
                raise ValueError(msg)

        return groups

    def __post_init__(self, resolutions_meters: dict[str, int]) -> None:
        """Check validity of modalities and further define non configurable attrs."""
        self._set_resolutions(resolutions_meters)

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

        if self.ref_input is not None:
            if any(name_mod not in self.rasters for name_mod in self.targets):
                msg = "All targets should be rasters when ref input is provided."
                raise ValueError(msg)
            if self.ref_input not in self.filter_inputs:
                msg = (
                    f"Invalid reference input {self.ref_input}."
                    f"Expected one of {self.filter_inputs}"
                )
                raise ValueError(msg)
        else:  # noqa: PLR5501
            if any(name_mod in self.rasters for name_mod in self.targets):
                msg = "None of the targets should be rasters when ref input is None."
                raise ValueError(msg)
