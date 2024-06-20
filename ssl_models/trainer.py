from pathlib import Path

import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from ssl_models.logger import ImageLogger, TensorBoardLogger


class SSLTrainer(pl.Trainer):
    def __init__(
        self,
        run,
        ssl_phase,
        opt,
        *args,
        **kwargs,
    ) -> None:
        self.ssl_phase = ssl_phase
        self.exp_dir = run.exp_dir
        self.exp_name = run.exp_name
        self.exp_version = self.get_hydra_timestamp()

        self.logged_images_per_epoch = run.logged_images_per_epoch

        self.base_lr = opt.base_lr
        self.wd = opt.wd
        self.b1 = opt.b1
        self.b2 = opt.b2

        self.loggers = self.configure_loggers()
        self.callbacks = self.configure_callbacks()
        super().__init__(
            logger=self.loggers,
            callbacks=self.callbacks,
            max_epochs=opt.max_epochs,
            *args,
            **kwargs,
        )

    def configure_loggers(self):
        self.tb_logger = TensorBoardLogger(
            ssl_phase=self.ssl_phase,
            save_dir=self.exp_dir,
            name=self.exp_name,
            version=self.exp_version,
            default_hp_metric=False,
        )
        return [self.tb_logger]

    def configure_callbacks(self):
        callbacks = []
        callbacks.append(TQDMProgressBar(refresh_rate=10))
        callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
            ),
        )
        callbacks.append(ModelCheckpoint(filename=self.ssl_phase + "-{epoch}"))
        callbacks.append(ImageLogger())
        return callbacks

    @staticmethod
    def get_hydra_timestamp():
        hydra_cfg = HydraConfig.get()
        return Path(hydra_cfg.runtime.output_dir).stem
