from logging import getLogger

from odeon.core.app_utils import Stages
from odeon.data import Input

from tests import TEST_DATA_DIR

logger = getLogger(__name__)

TRAIN_DATASET = TEST_DATA_DIR / "train.csv"
VAL_DATASET = TEST_DATA_DIR / "val.csv"
DEFAULT_PARAMS = {
    "input_fields": {
        "image": {"name": "image", "type": "raster", "dtype": "uint8"},
    },
    "input_file": TRAIN_DATASET,
    "root_dir": TEST_DATA_DIR,
}


def test_train_data_module():
    data_module = Input(fit_params=DEFAULT_PARAMS)
    data_module.setup(stage=str(Stages.FIT.value))
    data_loader = data_module.train_dataloader()
    n_cycle = 2
    for _ in range(n_cycle):
        for idx, batch in enumerate(data_loader):
            logger.info(idx)
            if int(idx) > n_cycle:
                logger.info("exit")
                break
            logger.info(f"id: {idx}, batch: {batch.keys()}")


def test_val_data():
    data_module = Input(validate_params={"input_file": VAL_DATASET, **DEFAULT_PARAMS})
    data_module.setup(stage=str(Stages.VALIDATE.value))
    data_loader = data_module.val_dataloader()
    n_cycle = 2
    for _ in range(n_cycle):
        for idx, batch in enumerate(data_loader):
            logger.info(idx)
            if int(idx) > n_cycle:
                logger.info("exit")
                break
            logger.info(f"id: {idx}, batch: {batch.keys()}")
