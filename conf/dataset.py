"""Dataset class."""

from hydra_zen import MISSING, builds, store

from conf.datasets.flair import FLAIRConfig
from conf.datasets.geopile import GeoPileConfig
from conf.datasets.hi_ucd import HiUCDConfig
from conf.datasets.pastis_hd import PASTISHDConfig
from conf.datasets.s2_naip import S2NAIPConfig
from conf.datasets.treesatai_ts import TreeSatAITSConfig

ALL_DATASETS = [
    "geopile",
    "flair",
    "pastis_hd",
    "treesatai_ts",
    "s2_naip",
    "hi_ucd",
]


class DatasetConfig:
    """Dataset config."""

    def __init__(
        self,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Assign arguments to config attributes."""
        self.__dict__ = kwargs
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check that input/target names are valid."""
        for name in self.datasets_pretrain + self.datasets_finetune:
            if name not in self.__dict__:
                msg = f"Invalid dataset name {name}. Not an attribute."
                raise ValueError(msg)


dataset_build = builds(
    DatasetConfig,
    root_dir=MISSING,
    datasets_pretrain=ALL_DATASETS,
    datasets_finetune=ALL_DATASETS,
    geopile=GeoPileConfig(),
    flair=FLAIRConfig(),
    pastis_hd=PASTISHDConfig(),
    treesatai_ts=TreeSatAITSConfig(),
    s2_naip=S2NAIPConfig(),
    hi_ucd=HiUCDConfig(),
)

dataset_store = store(group="dataset")
dataset_store(dataset_build, name="default_dataset")
