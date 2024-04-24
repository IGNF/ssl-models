import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor


class ImageLogger(Callback):
    """Log masked images, predicted images, and target images."""

    def __init__(
        self,
        every_n_batches: int = 10,
        samples_per_batch: int = 1,
    ) -> None:
        self.every_n_batches = every_n_batches
        self.samples_per_batch = samples_per_batch

    @staticmethod
    def to_numpy(x: Tensor):
        return x.detach().cpu().numpy().astype(np.float32)

    def on_batch_end(
        self,
        trainer: Trainer,
        outputs,
        batch,
        batch_idx,
        phase: str,
    ) -> None:
        for sample_idx, (masked, pred, cube) in enumerate(
            zip(outputs["masked"], outputs["pred"], batch["image"])
        ):
            if (
                batch_idx % self.every_n_batches == 0
                and sample_idx < self.samples_per_batch
            ):
                epoch = trainer.current_epoch
                prefix = (
                    f"{phase}/{batch_idx % 50}_{sample_idx}"
                    if self.samples_per_batch > 0
                    else f"{phase}/{batch_idx % 50}"
                )
                trainer.tb_logger._experiment.add_image(
                    f"{prefix}_input",
                    ImageLogger.to_numpy(masked),
                    epoch,
                )
                trainer.tb_logger._experiment.add_image(
                    f"{prefix}_pred",
                    ImageLogger.to_numpy(pred),
                    epoch,
                )
                trainer.tb_logger._experiment.add_image(
                    f"{prefix}_target",
                    ImageLogger.to_numpy(cube),
                    epoch,
                )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        self.on_batch_end(
            trainer,
            outputs,
            batch,
            batch_idx,
            phase="train",
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        self.on_batch_end(
            trainer,
            outputs,
            batch,
            batch_idx,
            phase="eval",
        )
