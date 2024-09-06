import itertools

import pytest
from odeon.core.app_utils import Stages

from ssl_models import LOGGER
from ssl_models.data.odeon_data import OdeonData
from tests import TEST_DATA_DIR


@pytest.mark.parametrize(
    "ssl_phase, stage, use_transform",
    itertools.product(
        ("pretrain", "probe", "finetune"),
        (Stages.FIT.value, Stages.VALIDATE.value),
        (True, False),
    ),
)
def test_odeon_data(ssl_phase, stage, use_transform):
    data_module = OdeonData(
        ssl_phase=ssl_phase,
        root_dir=TEST_DATA_DIR,
        batch_size=2,
        num_workers=1,
        use_transform=use_transform,
    )
    data_module.setup(stage=stage)
    if stage == Stages.FIT.value:
        data_loader = data_module.train_dataloader()
    else:
        data_loader = data_module.val_dataloader()
    num_cycles = 2
    for _ in range(num_cycles):
        for idx, batch in enumerate(data_loader):
            LOGGER.info(idx)
            if int(idx) > num_cycles:
                LOGGER.info("exit")
                break
            LOGGER.info(f"id: {idx}, batch: {batch.keys()}")


@pytest.mark.xfail
def test_odeon_should_fail():
    data_module = OdeonData(
        ssl_phase="",
        root_dir=TEST_DATA_DIR,
        batch_size=2,
        num_workers=1,
        use_transform=False,
    )
