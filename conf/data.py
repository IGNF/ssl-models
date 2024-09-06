"""Data conf."""

from dataclasses import dataclass

from hydra_zen import MISSING, store

from conf import pbuilds
from ssl_models.data.geopile_data import GeoPileData


@dataclass
class DataConfig:  # noqa: D101
    root_dir: str = MISSING
    use_transform: bool = True
    batch_size: int = 64
    num_workers: int = 32


data_build = pbuilds(GeoPileData, builds_bases=(DataConfig,))

data_store = store(group="data")
data_store(data_build, name="default_data")
