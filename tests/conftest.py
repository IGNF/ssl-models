import pytest

from conf.dataset.geopile import GeoPileConfig
from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig
from conf.opt import OptFinetuneConfig
from tests import TEST_DATA_DIR


@pytest.fixture()
def datasets_geopile():
    return DatasetsConfig(
        root_dir=TEST_DATA_DIR,
        geopile=GeoPileConfig(rel_dir=""),
        filter_pretrain=["geopile"],
        filter_finetune=["geopile"],
    )


@pytest.fixture()
def mask():
    return MaskConfig()


@pytest.fixture()
def opt_finetune():
    return OptFinetuneConfig()
