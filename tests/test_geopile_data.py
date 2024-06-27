import itertools

import pytest

from ssl_models import LOGGER
from ssl_models.geopile_data import GeoPileData
from tests import TEST_DATA_DIR


@pytest.mark.parametrize(
    "stage, ssl_phase, use_tranform",
    itertools.product(
        ("fit", "validate"),
        ("pretrain", "probe", "finetune"),
        (False, True),
    ),
)
def test_geopile_data(stage, ssl_phase, use_tranform):
    data_module = GeoPileData(
        ssl_phase=ssl_phase,
        root_dir=TEST_DATA_DIR,
        use_tranform=use_tranform,
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
