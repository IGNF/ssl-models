from dataclasses import dataclass

from hydra_zen import MISSING, builds, store

from ssl_models.geopile_data import GeoPileData

# ================== Data Conf ==================
data_store = store(group="data")


@dataclass
class DataConfig:
    root_dir: str = MISSING
    shuffle_val: bool = True
    use_tranform: bool = True
    batch_size: int = 64
    num_workers: int = 32


data_build = builds(GeoPileData, builds_bases=(DataConfig,))
data_store(data_build, name="default_data")
