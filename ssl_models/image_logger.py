import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor


class ImageLogger(Callback):
    """Log masked images, predicted images, and target images."""

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
        if trainer.global_rank != 0:
            return
        match phase:
            case "train":
                num_batches_per_epoch = len(trainer.train_dataloader)
            case "eval":
                num_batches_per_epoch = len(trainer.val_dataloaders)
            case "test":
                num_batches_per_epoch = len(trainer.test_dataloaders)
        log_every_n_batches = round(
            num_batches_per_epoch / trainer.logged_images_per_epoch
        )
        if batch_idx % log_every_n_batches:
            return

        epoch = trainer.current_epoch
        prefix_idx = batch_idx // log_every_n_batches
        trainer.tb_logger._experiment.add_image(
            f"{phase}/{prefix_idx}_input",
            ImageLogger.to_numpy(outputs["masked"][0]),
            epoch,
        )
        trainer.tb_logger._experiment.add_image(
            f"{phase}/{prefix_idx}_pred",
            ImageLogger.to_numpy(outputs["pred"][0]),
            epoch,
        )
        trainer.tb_logger._experiment.add_image(
            f"{phase}/{prefix_idx}_target",
            ImageLogger.to_numpy(batch["image"][0]),
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
