import pytest

from ssl_models.conf.dataset.bdfv2 import BDFV2Config
from ssl_models.conf.dataset.circa import CircaConfig
from ssl_models.conf.dataset.flair import FLAIRConfig
from ssl_models.conf.dataset.geopile import GeoPileConfig
from ssl_models.conf.dataset.hi_ucd import HiUCDConfig
from ssl_models.conf.dataset.pastis_hd import PASTISHDConfig
from ssl_models.conf.dataset.s2_naip import S2NAIPConfig
from ssl_models.conf.dataset.treesatai_ts import TreeSatAITSConfig
from ssl_models.conf.mask import MiMMaskConfig
from ssl_models.conf.opt import OptFinetuneConfig
from tests import TEST_DATA_DIR


@pytest.fixture()
def dataset_treesat():
    return TreeSatAITSConfig(rel_dir="")


@pytest.fixture()
def mask():
    return MiMMaskConfig()


@pytest.fixture()
def opt_finetune():
    return OptFinetuneConfig()
