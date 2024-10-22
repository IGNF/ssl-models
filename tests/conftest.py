import pytest

from conf.dataset.geopile import GeoPileConfig
from conf.datasets import DatasetsConfig
from tests import TEST_DATA_DIR


@pytest.fixture()
def datasets_geopile():
    return DatasetsConfig(
        root_dir=TEST_DATA_DIR,
        geopile=GeoPileConfig(rel_dir=""),
        filter_allowed=["geopile"],
        filter_pretrain=["geopile"],
        filter_finetune=["geopile"],
    )
