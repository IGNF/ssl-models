import itertools

import pytest

from ssl_models import LOGGER
from ssl_models.train.data import SSLDataModule
from tests import TEST_DATA_DIR


@pytest.mark.parametrize(
    "stage, ssl_phase, use_transform, random_dates, random_crop",
    itertools.product(
        ("fit", "validate"),
        ("pretrain", "probe", "finetune"),
        (True, False),
        (True, False),
        (True, False),
    ),
)
def test_data(
    dataset_treesat,
    opt_finetune,
    stage,
    ssl_phase,
    use_transform,
    random_dates,
    random_crop,
):
    data_module = SSLDataModule(
        dataset=dataset_treesat,
        root_dir=TEST_DATA_DIR,
        ssl_phase=ssl_phase,
        opt=opt_finetune,
        num_workers=1,
        use_transform=use_transform,
        random_dates=random_dates,
        random_crop=random_crop,
    )
    data_module.setup(stage=stage)
    data_loader = data_module.train_dataloader()
    num_cycles = 2
    for _ in range(num_cycles):
        for idx, batch in enumerate(data_loader):
            LOGGER.info(idx)
            if int(idx) > num_cycles:
                LOGGER.info("exit")
                break
            LOGGER.info(f"id: {idx}, batch: {batch.keys()}")
