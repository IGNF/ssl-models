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
        name_dataset: str,
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
            and (not is_dataclass(val_arg) or name_arg == name_dataset)
        }
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check validity of dataset name and further define non configurable attrs."""
        if self.name_dataset not in self.__dict__:
            msg = f"Invalid dataset name {self.name_dataset}. Not an attribute."
            raise ValueError(msg)

        self.dataset = getattr(self, self.name_dataset)
        self.dataset_class = self.dataset.dataset_class


dataset_build = builds(
    DatasetsConfig,
    root_dir=MISSING,
    name_dataset=MISSING,
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
