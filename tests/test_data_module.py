from odeon.core.app_utils import Stages
from odeon.data import Input

from ssl_models import LOGGER
from ssl_models.data_module import DEFAULT_PARAMS
from tests import TEST_DATA_DIR

TRAIN_DATASET = TEST_DATA_DIR / "train.csv"
VAL_DATASET = TEST_DATA_DIR / "val.csv"


def test_train_data_module():
    data_module = Input(
        fit_params={
            **DEFAULT_PARAMS,
            "input_file": TRAIN_DATASET,
            "root_dir": TEST_DATA_DIR,
        }
    )
    data_module.setup(stage=str(Stages.FIT.value))
    data_loader = data_module.train_dataloader()
    n_cycle = 2
    for _ in range(n_cycle):
        for idx, batch in enumerate(data_loader):
            LOGGER.info(idx)
            if int(idx) > n_cycle:
                LOGGER.info("exit")
                break
            LOGGER.info(f"id: {idx}, batch: {batch.keys()}")


def test_val_data():
    data_module = Input(
        validate_params={
            **DEFAULT_PARAMS,
            "input_file": VAL_DATASET,
            "root_dir": TEST_DATA_DIR,
        }
    )
    data_module.setup(stage=str(Stages.VALIDATE.value))
    data_loader = data_module.val_dataloader()
    n_cycle = 2
    for _ in range(n_cycle):
        for idx, batch in enumerate(data_loader):
            LOGGER.info(idx)
            if int(idx) > n_cycle:
                LOGGER.info("exit")
                break
            LOGGER.info(f"id: {idx}, batch: {batch.keys()}")
