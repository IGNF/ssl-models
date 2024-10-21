"""Dataset class."""

from hydra_zen import MISSING, builds, store

from conf.dataset.flair import FLAIRConfig
from conf.dataset.geopile import GeoPileConfig
from conf.dataset.hi_ucd import HiUCDConfig
from conf.dataset.pastis_hd import PASTISHDConfig
from conf.dataset.s2_naip import S2NAIPConfig
from conf.dataset.treesatai_ts import TreeSatAITSConfig

ALLOWED_DATASETS = [
    "flair",
    "geopile",
    "hi_ucd",
    "pastis_hd",
    "s2_naip",
    "treesatai_ts",
]


# It does not seems possible to have nested dataclasses at all config levels,
# so we do not use a dataclass at the top level.
class DatasetsConfig:
    """Dataset config."""

    def __init__(
        self,
        root_dir: str,
        filter_pretrain: list[str],
        filter_finetune: list[str],
        geopile: GeoPileConfig,
        flair: FLAIRConfig,
        pastis_hd: PASTISHDConfig,
        treesatai_ts: TreeSatAITSConfig,
        s2_naip: S2NAIPConfig,
        hi_ucd: HiUCDConfig,
    ) -> None:
        """Assign args to config attributes."""
        self.root_dir = root_dir
        self.filter_pretrain = filter_pretrain
        self.filter_finetune = filter_finetune
        self.geopile = geopile
        self.flair = flair
        self.pastis_hd = pastis_hd
        self.treesatai_ts = treesatai_ts
        self.s2_naip = s2_naip
        self.hi_ucd = hi_ucd
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check validity of dataset names and further define non configurable attrs."""
        for name_dataset in self.filter_pretrain + self.filter_finetune:
            if name_dataset not in self.__dict__:
                msg = f"Invalid dataset name {name_dataset}. Not an attribute."
                raise ValueError(msg)
        self.allowed = {
            name_dataset: getattr(self, name_dataset)
            for name_dataset in ALLOWED_DATASETS
        }
        self.pretrain = {
            name_dataset: getattr(self, name_dataset)
            for name_dataset in self.filter_pretrain
        }
        self.finetune = {
            name_dataset: getattr(self, name_dataset)
            for name_dataset in self.filter_finetune
        }

        # temporary restriction to single datasets with single modality and single date
        names_dataset = self.filter_pretrain
        name_dataset = self.filter_pretrain[0]
        dataset = self.pretrain[name_dataset]
        mod_name = dataset.filter_inputs[0]
        mod = dataset.inputs[mod_name]

        if len(names_dataset) > 1:
            msg = "Multi-dataset pretraining/finetuning not yet implemented."
            raise NotImplementedError(msg)

        if len(dataset.filter_inputs) != 1:
            msg = "Multi-modality pretraining/finetuning not yet implemented."
            raise NotImplementedError(msg)

        if mod.num_dates != 1:
            msg = "Multi-dates pretraining/finetuning not yet implemented."
            raise NotImplementedError(msg)

        self.dataset_class = dataset.dataset_class
        self.dataset = dataset
        self.mod_name = mod_name
        self.mod = mod


dataset_build = builds(
    DatasetsConfig,
    root_dir=MISSING,
    filter_pretrain=ALLOWED_DATASETS,
    filter_finetune=ALLOWED_DATASETS,
    geopile=GeoPileConfig(),
    flair=FLAIRConfig(),
    pastis_hd=PASTISHDConfig(),
    treesatai_ts=TreeSatAITSConfig(),
    s2_naip=S2NAIPConfig(),
    hi_ucd=HiUCDConfig(),
)

dataset_store = store(group="datasets")
dataset_store(dataset_build, name="default_datasets")
