"""Data conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class DataConfig:  # noqa: D101
    use_transform: bool = True
    batch_size: int = 64
    num_workers: int = 12


data_store = store(group="data")
data_store(DataConfig, name="default_data")
