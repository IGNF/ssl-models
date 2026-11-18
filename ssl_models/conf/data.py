"""Data conf."""

from dataclasses import dataclass

from hydra_zen import MISSING, store


@store(group="data", name="default_data")
@dataclass
class DataConfig:
    """Data reading, used by the data module and datasets only.

    `random_crop` and `random_dates` are read-time augmentations (train stage only),
    GPU augmentations being set in the augment config.
    """

    root_dir: str = MISSING
    num_workers: int = 12
    random_crop: bool = True
    random_dates: bool = True
