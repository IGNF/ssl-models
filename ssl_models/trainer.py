import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


class SSLTrainer(pl.Trainer):
    def __init__(
        self,
        exp_dir: str,
        exp_name: str,
        exp_version: int | None = None,
        accelerator: str = "auto",
        devices: str = "auto",
        strategy: str = "ddp",
        precision: str = "bf16-mixed",
        log_every_n_steps: int = 1,
        max_epochs: int = 1000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            log_every_n_steps=log_every_n_steps,
            max_epochs=max_epochs,
            *args,
            **kwargs,
        )

        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.exp_version = exp_version

        self.loggers = self.configure_loggers()
        self.callbacks = self.configure_callbacks()

    def configure_loggers(self):
        # csv logger
        self.csv_logger = CSVLogger(
            save_dir=self.exp_dir, name=self.exp_name, version=self.exp_version
        )

        # tensorboard logger
        self.tb_logger = TensorBoardLogger(
            save_dir=self.exp_dir,
            name=self.exp_name,
            version=self.exp_version,
            default_hp_metric=False,
        )

        return [self.csv_logger, self.tb_logger]

    def configure_callbacks(self):
        callbacks = []

        # Progress bar
        callbacks.append(TQDMProgressBar(refresh_rate=10))

        # Sauvegarde des checkpoints pour la meilleure mIoU
        callbacks.append(ModelCheckpoint(filename="{epoch}"))

        return callbacks

    def fit(
        self,
        model: pl.LightningModule,
        *args,
        **kwargs,
    ) -> None:
        model.tb_logger = self.tb_logger
        super().fit(
            model,
            *args,
            **kwargs,
        )
