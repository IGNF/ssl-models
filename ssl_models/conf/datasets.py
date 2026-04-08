"""Dataset class."""

from dataclasses import is_dataclass

from hydra_zen import MISSING, builds, store

from ssl_models.conf.dataset.bdfv2 import BDFV2Config
from ssl_models.conf.dataset.circa import CircaConfig
from ssl_models.conf.dataset.flair import FLAIRConfig
from ssl_models.conf.dataset.geopile import GeoPileConfig
from ssl_models.conf.dataset.hi_ucd import HiUCDConfig
from ssl_models.conf.dataset.pastis_hd import PASTISHDConfig
from ssl_models.conf.dataset.s2_naip import S2NAIPConfig
from ssl_models.conf.dataset.treesatai_ts import TreeSatAITSConfig


# It does not seems possible to have nested dataclasses at all config levels,
# so we do not use a dataclass at the top level.
class DatasetsConfig:
    """Dataset config."""

    def __init__(
        # ruff: noqa: B008
        self,
        root_dir: str,
        name_datasets: list[str],
        sample_datasets: list[float] | None = None,
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
        self.__dict__ = {
            name_arg: val_arg
            for name_arg, val_arg in locals().items()
            if val_arg is not self
            and (not is_dataclass(val_arg) or name_arg in name_datasets)
        }
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check validity of dataset name and further define non configurable attrs."""
        if self.sample_datasets is None:
            self.sample_datasets = [1.0] * len(self.name_datasets)

        if len(self.name_datasets) != len(self.sample_datasets):
            msg = "Number of datasets and sampling weights are inconsistent."
            raise ValueError(msg)

        self.datasets = {}
        self.sample = {}
        for name_dataset, sample_dataset in zip(
            self.name_datasets,
            self.sample_datasets,
        ):
            if name_dataset not in self.__dict__:
                msg = f"Invalid dataset name {name_dataset}. Not an attribute."
                raise ValueError(msg)

            self.datasets[name_dataset] = getattr(self, name_dataset)
            self.sample[name_dataset] = sample_dataset


dataset_build = builds(
    DatasetsConfig,
    root_dir=MISSING,
    name_datasets=MISSING,
    sample_datasets=None,
    geopile=GeoPileConfig(),
    flair=FLAIRConfig(),
    pastis_hd=PASTISHDConfig(),
    treesatai_ts=TreeSatAITSConfig(),
    s2_naip=S2NAIPConfig(),
    hi_ucd=HiUCDConfig(),
    circa=CircaConfig(),
    bdfv2=BDFV2Config(),
)

dataset_store = store(group="datasets")
dataset_store(dataset_build, name="default_datasets")
