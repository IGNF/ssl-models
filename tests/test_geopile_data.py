import itertools

import pytest

from ssl_models import LOGGER
from ssl_models.data.geopile import GeoPileData
from tests import TEST_DATA_DIR


@pytest.mark.parametrize(
    "stage, ssl_phase, use_transform",
    itertools.product(
        ("fit", "validate"),
        ("pretrain", "probe", "finetune"),
        (False, True),
    ),
)
def test_geopile_data(stage, ssl_phase, use_transform):
    data_module = GeoPileData(
        ssl_phase=ssl_phase,
        root_dir=TEST_DATA_DIR,
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
