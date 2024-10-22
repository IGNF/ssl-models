import itertools

import pytest

from conf.dataset.geopile import GeoPileConfig
from conf.datasets import DatasetsConfig
from ssl_models import LOGGER
from ssl_models.train.data import SSLDataModule
from tests import TEST_DATA_DIR


@pytest.mark.parametrize(
    "stage, ssl_phase, use_transform",
    itertools.product(
        ("fit", "validate"),
        ("pretrain", "probe", "finetune"),
        (False, True),
    ),
)
def test_data(datasets_geopile, stage, ssl_phase, use_transform):
    data_module = SSLDataModule(
        datasets=datasets_geopile,
        ssl_phase=ssl_phase,
        batch_size=2,
        num_workers=1,
        use_transform=use_transform,
    )
    data_module.setup(stage=stage)
    data_loader = data_module.val_dataloader()
    num_cycles = 2
    for _ in range(num_cycles):
        for idx, batch in enumerate(data_loader):
            LOGGER.info(idx)
            if int(idx) > num_cycles:
                LOGGER.info("exit")
                break
            LOGGER.info(f"id: {idx}, batch: {batch.keys()}")
