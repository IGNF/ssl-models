from logging import getLogger

from pytorch_lightning.callbacks import BasePredictionWriter

logger = getLogger(__name__)


class PredictWritterCallback(BasePredictionWriter):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer):
        logger.info("Starting")
