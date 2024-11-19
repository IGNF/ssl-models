"""Logger module."""

from typing import Literal

import cv2
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor


class TensorBoardLogger(TensorBoardLogger):
    """Custom tensorboard logger."""

    def __init__(
        self,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        save_dir: str,
        name: str,
        version: str,
        default_hp_metric: bool,
    ) -> None:
        self.ssl_phase = ssl_phase
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            default_hp_metric=default_hp_metric,
        )

    def add_image(self, tag: str, image: Tensor, epoch: int) -> None:
        """Add image to tensorboard logger."""
        self._experiment.add_image(tag, image, epoch)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics, appending ssl phase info to epoch logging."""
        if "epoch" in metrics:
            metrics[f"{self.ssl_phase}_epoch"] = metrics["epoch"]
            metrics.pop("epoch", None)
        super().log_metrics(metrics, step)


class ImageLogger(Callback):
    """Log masked images, predicted images, and target images."""

    @staticmethod
    def to_numpy(x: Tensor) -> Tensor:
        """Move to CPU and convert to numpy."""
        return x.detach().cpu().numpy().astype(np.float32)[:3]

    @staticmethod
    def should_log(trainer: Trainer, outputs: dict) -> bool:
        """Check whether images should be logged."""
        return trainer.global_rank == 0 and outputs["pred"] is not None

    def on_batch_end(
        self,
        trainer: Trainer,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> None:
        """Log images."""
        match stage:
            case "train":
                num_batches_per_epoch = len(trainer.train_dataloader)
            case "val":
                num_batches_per_epoch = len(trainer.val_dataloaders)
            case "test":
                num_batches_per_epoch = len(trainer.test_dataloaders)
        mod_name = trainer.mod_name
        log_every_n_batches = round(
            num_batches_per_epoch / trainer.logged_images_per_epoch,
        )
        if batch_idx % log_every_n_batches:
            return

        epoch = trainer.current_epoch
        prefix_idx = batch_idx // log_every_n_batches

        match trainer.ssl_phase:
            case "pretrain":
                trainer.tb_logger.add_image(
                    f"{trainer.ssl_phase}_{stage}/{prefix_idx}_input",
                    ImageLogger.to_numpy(outputs["masked"][0]),
                    epoch,
                )
                trainer.tb_logger.add_image(
                    f"{trainer.ssl_phase}_{stage}/{prefix_idx}_pred",
                    ImageLogger.to_numpy(outputs["pred"][0]),
                    epoch,
                )
                trainer.tb_logger.add_image(
                    f"{trainer.ssl_phase}_{stage}/{prefix_idx}_target",
                    ImageLogger.to_numpy(batch[mod_name][0]),
                    epoch,
                )
            case "finetune" | "probe":
                for name_target, pred_target in outputs["pred"].items():
                    mod_ref = trainer.mod_name

                    input_ = ImageLogger.to_numpy(batch[mod_ref][0])
                    pred = ImageLogger.to_numpy(pred_target[0])
                    mask = ImageLogger.to_numpy(batch[name_target][0])

                    if input_.shape[-1] > mask.shape[-1]:
                        pred = cv2.resize(
                            pred.squeeze(),
                            (input_.shape[-1], input_.shape[-1]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        mask = cv2.resize(
                            mask.squeeze(),
                            (input_.shape[-1], input_.shape[-1]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    if trainer.targets[name_target].type_target == "segment":
                        colormap = np.array(trainer.targets[name_target].colormap)
                        pred = colormap.T[:, pred.astype(np.uint8)].squeeze()
                        mask = colormap.T[:, mask.astype(np.uint8)].squeeze()

                    trainer.tb_logger.add_image(
                        f"{trainer.ssl_phase}_{name_target}_{stage}/{prefix_idx}_input",
                        input_,
                        epoch,
                    )

                    trainer.tb_logger.add_image(
                        f"{trainer.ssl_phase}_{name_target}_{stage}/{prefix_idx}_pred",
                        pred,
                        epoch,
                    )

                    trainer.tb_logger.add_image(
                        f"{trainer.ssl_phase}_{name_target}_{stage}/{prefix_idx}_target",
                        mask,
                        epoch,
                    )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Log train images."""
        if ImageLogger.should_log(trainer, outputs):
            self.on_batch_end(
                trainer,
                outputs,
                batch,
                batch_idx,
                stage="train",
            )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Log validation images."""
        if ImageLogger.should_log(trainer, outputs):
            self.on_batch_end(
                trainer,
                outputs,
                batch,
                batch_idx,
                stage="val",
            )
