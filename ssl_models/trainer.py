import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from ssl_models.image_logger import ImageLogger


class SSLTrainer(pl.Trainer):
    def __init__(  # noqa: PLR0913
        self,
        exp_dir: str,
        exp_name: str,
        exp_version: int | None = None,
        use_clearml: bool = False,
        clearml_project_name: str = "ssl",
        logged_images_per_epoch: int = 50,
        base_lr: float = 1e-3,
        wd: float = 0.01,
        b1: float = 0.9,
        b2: float = 0.99,
        accelerator: str = "auto",
        devices: str = "auto",
        strategy: str = "ddp",
        precision: str = "bf16-mixed",
        log_every_n_steps: int = 1,
        max_epochs: int = 1000,
        logger: list = [],
        callbacks: list = [],
        *args,
        **kwargs,
    ) -> None:
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.exp_version = exp_version
        self.logged_images_per_epoch = logged_images_per_epoch
        self.base_lr = base_lr
        self.wd = wd
        self.b1 = b1
        self.b2 = b2

        self.loggers = logger + self.configure_loggers()
        self.callbacks = callbacks + self.configure_callbacks()
        super().__init__(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            log_every_n_steps=log_every_n_steps,
            max_epochs=max_epochs,
            logger=self.loggers,
            callbacks=self.callbacks,
            *args,
            **kwargs,
        )
        if use_clearml and self.global_rank == 0:
            Task.init(
                project_name=clearml_project_name,
                task_name=self.exp_name,
                tags=[],
            )

    def configure_loggers(self):
        self.tb_logger = TensorBoardLogger(
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
            )
        )
        callbacks.append(ModelCheckpoint(filename="{epoch}"))
        callbacks.append(ImageLogger())
        return callbacks
