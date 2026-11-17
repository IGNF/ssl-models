"""Data conf."""

from dataclasses import dataclass

from hydra_zen import MISSING, store


@store(group="data", name="default_data")
@dataclass
class DataConfig:  # noqa: D101
    root_dir: str = MISSING
    use_transform: bool = True
    random_dates: bool = True
    random_crop: bool = True
    num_workers: int = 12
