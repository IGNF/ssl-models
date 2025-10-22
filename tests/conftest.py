import pytest

from ssl_models.conf.dataset.bdfv2 import BDFV2Config
from ssl_models.conf.dataset.circa import CircaConfig
from ssl_models.conf.dataset.flair import FLAIRConfig
from ssl_models.conf.dataset.geopile import GeoPileConfig
from ssl_models.conf.dataset.hi_ucd import HiUCDConfig
from ssl_models.conf.dataset.pastis_hd import PASTISHDConfig
from ssl_models.conf.dataset.s2_naip import S2NAIPConfig
from ssl_models.conf.dataset.treesatai_ts import TreeSatAITSConfig
from ssl_models.conf.datasets import DatasetsConfig
from ssl_models.conf.mask import MaskConfig
from ssl_models.conf.opt import OptFinetuneConfig
from tests import TEST_DATA_DIR


@pytest.fixture()
def datasets_treesat():
    return DatasetsConfig(
        treesatai_ts=TreeSatAITSConfig(rel_dir=""),
        root_dir=TEST_DATA_DIR,
        geopile=GeoPileConfig(),
        flair=FLAIRConfig(),
        pastis_hd=PASTISHDConfig(),
        s2_naip=S2NAIPConfig(),
        hi_ucd=HiUCDConfig(),
        circa=CircaConfig(),
        bdfv2=BDFV2Config(),
        filter_pretrain=["treesatai_ts"],
        filter_finetune=["treesatai_ts"],
    )


@pytest.fixture()
def mask():
    return MaskConfig()


@pytest.fixture()
def opt_finetune():
    return OptFinetuneConfig()
