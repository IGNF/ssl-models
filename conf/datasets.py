"""Dataset class."""

from hydra_zen import MISSING, builds, store

from conf.dataset.flair import FLAIRConfig
from conf.dataset.geopile import GeoPileConfig
from conf.dataset.hi_ucd import HiUCDConfig
from conf.dataset.pastis_hd import PASTISHDConfig
from conf.dataset.s2_naip import S2NAIPConfig
from conf.dataset.treesatai_ts import TreeSatAITSConfig
from ssl_models.dataset.flair import FLAIRDataset
from ssl_models.dataset.geopile import GeoPileDataset
from ssl_models.dataset.hi_ucd import HiUCDDataset
from ssl_models.dataset.pastis_hd import PASTISHDDataset
from ssl_models.dataset.s2_naip import S2NAIPDataset
from ssl_models.dataset.treesatai_ts import TreeSatAITSDataset

DATASET_MAP = {
    "flair": FLAIRDataset,
    "geopile": GeoPileDataset,
    "hi_ucd": HiUCDDataset,
    "pastis_hd": PASTISHDDataset,
    "s2_naip": S2NAIPDataset,
    "treesatai_ts": TreeSatAITSDataset,
}
ALLOWED_DATASETS = list(DATASET_MAP.keys())


class DatasetsConfig:
    """Dataset config."""

    def __init__(
        self,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Assign arguments to config attributes."""
        self.__dict__ = kwargs
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check that dataset names are valid."""
        for name_dataset in self.filter_pretrain + self.filter_finetune:
            if name_dataset not in self.__dict__:
                msg = f"Invalid dataset name {name_dataset}. Not an attribute."
                raise ValueError(msg)

        # temporary restriction to single datasets with single modality and single date
        names_dataset = set(self.filter_pretrain + self.filter_finetune)
        name_dataset = next(names_dataset)
        dataset = getattr(self, name_dataset)
        mod = getattr(dataset, next(dataset.filter_input))

        if len(names_dataset) > 1:
            msg = "Multi-dataset pretraining/finetuning not yet implemented."
            raise NotImplementedError(msg)

        if len(dataset.filter_input) != 1:
            msg = "Multi-modality pretraining/finetuning not yet implemented."
            raise NotImplementedError(msg)

        if len(mod.num_dates) != 1:
            msg = "Multi-dates pretraining/finetuning not yet implemented."
            raise NotImplementedError(msg)

        self.dataset_class = DATASET_MAP[name_dataset]
        self.dataset = dataset
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
