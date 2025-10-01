# ruff: noqa: B008
"""Dataset class."""

from hydra_zen import MISSING, builds, store

from conf.dataset.bdfv2 import BDFV2Config
from conf.dataset.circa import CircaConfig
from conf.dataset.flair import FLAIRConfig
from conf.dataset.geopile import GeoPileConfig
from conf.dataset.hi_ucd import HiUCDConfig
from conf.dataset.pastis_hd import PASTISHDConfig
from conf.dataset.s2_naip import S2NAIPConfig
from conf.dataset.treesatai_ts import TreeSatAITSConfig

ALL_DATASETS = [
    "flair",
    "geopile",
    "hi_ucd",
    "pastis_hd",
    "s2_naip",
    "treesatai_ts",
    "circa",
    "bdfv2",
]


# It does not seems possible to have nested dataclasses at all config levels,
# so we do not use a dataclass at the top level.
class DatasetsConfig:
    """Dataset config."""

    def __init__(
        self,
        root_dir: str,
        filter_allowed: list[str] = ALL_DATASETS,
        filter_pretrain: list[str] = ALL_DATASETS,
        filter_finetune: list[str] = ALL_DATASETS,
        geopile: GeoPileConfig = GeoPileConfig(),
        flair: FLAIRConfig = FLAIRConfig(),
        pastis_hd: PASTISHDConfig = PASTISHDConfig(),
        treesatai_ts: TreeSatAITSConfig = TreeSatAITSConfig(),
        s2_naip: S2NAIPConfig = S2NAIPConfig(),
        hi_ucd: HiUCDConfig = HiUCDConfig(),
        circa: CircaConfig = CircaConfig(),
        bdfv2: BDFV2Config = BDFV2Config(),
    ) -> None:
        """Assign args to config attributes."""
        self.kwargs = {
            name_arg: val_arg
            for name_arg, val_arg in locals().items()
            if not isinstance(val_arg, DatasetsConfig)
        }
        for name_arg, val_arg in self.kwargs.items():
            self.__dict__[name_arg] = val_arg

        self.__post_init__()

    def __post_init__(self) -> None:
        """Check validity of dataset names and further define non configurable attrs."""
        for name_dataset in self.filter_pretrain + self.filter_finetune:
            if name_dataset not in self.__dict__:
                msg = f"Invalid dataset name {name_dataset}. Not an attribute."
                raise ValueError(msg)
        self.allowed = {
            name_dataset: getattr(self, name_dataset)
            for name_dataset in self.filter_allowed
        }
        self.pretrain = {
            name_dataset: getattr(self, name_dataset)
            for name_dataset in self.filter_pretrain
        }
        self.finetune = {
            name_dataset: getattr(self, name_dataset)
            for name_dataset in self.filter_finetune
        }

        # temporary restriction to single datasets
        names_dataset = self.filter_pretrain
        name_dataset = self.filter_pretrain[0]
        dataset = self.pretrain[name_dataset]

        if len(names_dataset) > 1:
            msg = "Multi-dataset pretraining/finetuning not implemented yet."
            raise NotImplementedError(msg)

        self.dataset_class = dataset.dataset_class
        self.dataset = dataset


dataset_build = builds(
    DatasetsConfig,
    root_dir=MISSING,
)

dataset_store = store(group="datasets")
dataset_store(dataset_build, name="default_datasets")
