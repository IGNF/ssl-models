from ssl_models import LOGGER
from ssl_models.geopile_data import GeoPileData
from tests import TEST_DATA_DIR


def test_train_geopile_data():
    data_module = GeoPileData(root_dir=TEST_DATA_DIR)
    data_module.setup(stage="fit")
    data_loader = data_module.val_dataloader()
    num_cycles = 2
    for _ in range(num_cycles):
        for idx, batch in enumerate(data_loader):
            LOGGER.info(idx)
            if int(idx) > num_cycles:
                LOGGER.info("exit")
                break
            LOGGER.info(f"id: {idx}, batch: {batch.keys()}")


def test_val_geopile_data():
    data_module = GeoPileData(root_dir=TEST_DATA_DIR)
    data_module.setup(stage="validate")
    data_loader = data_module.val_dataloader()
    num_cycles = 2
    for _ in range(num_cycles):
        for idx, batch in enumerate(data_loader):
            LOGGER.info(idx)
            if int(idx) > num_cycles:
                LOGGER.info("exit")
                break
            LOGGER.info(f"id: {idx}, batch: {batch.keys()}")
