"""Trainer module."""

from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers.logger import Logger

from conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig
from conf.run import RunConfig
from ssl_models.hydra_utils import get_hydra_timestamp
from ssl_models.train.logger import ImageLogger, TensorBoardLogger


class SSLTrainer(pl.Trainer):
    """SSL trainer."""

    def __init__(
        self,
        run: RunConfig,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        opt: OptPretrainConfig | OptProbeConfig | OptFinetuneConfig,
        accelerator: str,
        devices: str | int,
        strategy: str,
        precision: str,
    ) -> None:
        self.ssl_phase = ssl_phase
        self.exp_dir = run.exp_dir
        self.exp_name = run.exp_name
        self.exp_version = get_hydra_timestamp()

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
            max_epochs=opt.epochs,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
        )

    def configure_loggers(self) -> list[Logger]:
        """Configure loggers."""
        self.tb_logger = TensorBoardLogger(
            ssl_phase=self.ssl_phase,
            save_dir=self.exp_dir,
            name=self.exp_name,
            version=self.exp_version,
            default_hp_metric=False,
        )
        return [self.tb_logger]

    def configure_callbacks(self) -> list[Callback]:
        """Configure callbacks."""
        callbacks = []
        callbacks.append(TQDMProgressBar(refresh_rate=10))
        callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
            ),
        )
        callbacks.append(
            ModelCheckpoint(
                filename=self.ssl_phase + "-{epoch}",
                save_weights_only=True,
            ),
        )
        callbacks.append(ImageLogger())
        return callbacks
