"""Data conf."""

from dataclasses import dataclass

from hydra_zen import MISSING, builds, store

from ssl_models.geopile_data import GeoPileData


@dataclass
class DataConfig:  # noqa: D101
    root_dir: str = MISSING
    use_tranform: bool = True
    batch_size: int = 64
    num_workers: int = 32


data_build = builds(GeoPileData, builds_bases=(DataConfig,))

data_store = store(group="data")
data_store(data_build, name="default_data")
