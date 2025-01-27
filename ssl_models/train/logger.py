"""Logger module."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor

BANDS_RGB = 3


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

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)

    @staticmethod
    def to_numpy(x: Tensor) -> np.ndarray:
        """Move to CPU and convert to numpy."""
        x_npy = x.detach().cpu().numpy().astype(np.float32)
        return (
            x_npy[:3]
            if x_npy.shape[0] >= BANDS_RGB
            else x_npy.mean(axis=0, keepdims=True)
        )

    @staticmethod
    def should_log(trainer: Trainer) -> bool:
        """Check whether images should be logged."""
        return trainer.global_rank == 0

    def on_batch_end(
        self,
        trainer: Trainer,
        outputs: dict[str, dict],
        batch: dict[str, Tensor],  # noqa: ARG002
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
        log_every_n_batches = round(
            num_batches_per_epoch / trainer.logged_images_per_epoch,
        )
        if batch_idx % log_every_n_batches:
            return
        epoch = trainer.current_epoch
        prefix_idx = batch_idx // log_every_n_batches

        for log_title, log_img in outputs["log_inputs"].items():
            trainer.tb_logger.add_image(
                log_title.replace("/", f"/{prefix_idx}"),
                ImageLogger.to_numpy(log_img),
                epoch,
            )
        for log_title, log_img in outputs["log_preds"].items():
            trainer.tb_logger.add_image(
                log_title.replace("/", f"/{prefix_idx}"),
                ImageLogger.to_numpy(log_img),
                epoch,
            )
        for log_title, log_img in outputs["log_targets"].items():
            trainer.tb_logger.add_image(
                log_title.replace("/", f"/{prefix_idx}"),
                ImageLogger.to_numpy(log_img),
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
        if ImageLogger.should_log(trainer):
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
        if ImageLogger.should_log(trainer):
            self.on_batch_end(
                trainer,
                outputs,
                batch,
                batch_idx,
                stage="val",
            )


class ConfusionMatrixLogger(Callback):
    """Log confusion matrix."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)

    def on_epoch_end(
        self,
        trainer: Trainer,
        metrics: dict[str, dict[str, Tensor]],
    ) -> None:
        for name_target, metrics_target in metrics.items():
            if "confusion_matrix" in metrics_target:
                cm = metrics_target["confusion_matrix"].cpu().numpy()
                cm_df = pd.DataFrame(cm)

                if cm.shape[0] <= 10:  # noqa: PLR2004
                    figsize = (10, 7)
                elif cm.shape[0] > 10 and cm.shape[0] <= 16:  # noqa: PLR2004
                    figsize = (12, 9)
                elif cm.shape[0] > 16 and cm.shape[0] <= 20:  # noqa: PLR2004
                    figsize = (16, 11)
                else:
                    figsize = (21, 16)

                fig, ax = plt.subplots(figsize=figsize)
                sns.heatmap(cm_df, annot=True, fmt=".0f")
                log_cm_title = f"{name_target} Confusion matrix"
                plt.suptitle(log_cm_title)
                fig.savefig(str(Path(trainer.tb_logger.log_dir)))

                cm_norm_df = pd.DataFrame(
                    np.divide(
                        cm.astype(np.float32),
                        np.sum(cm, axis=1)[:, np.newaxis],
                        out=np.zeros_like(cm.astype(np.float32)),
                        where=(np.sum(cm, axis=1)[:, np.newaxis] != 0),
                    ),
                )
                fig, ax = plt.subplots(figsize=figsize)
                sns.heatmap(cm_norm_df, annot=True, fmt=".3f")
                log_cm_title = f"{name_target} Confusion matrix normalized"
                plt.suptitle(log_cm_title)
                fig.savefig(str(Path(trainer.tb_logger.log_dir)))

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if pl_module.metrics and trainer.ssl_phase == "finetune":
            metrics_val = {
                k: pl_module.metrics[k] for k in pl_module.metrics if "val" in k
            }
            if metrics_val:
                self.on_epoch_end(trainer, metrics_val)

    @rank_zero_only
    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if pl_module.metrics and trainer.ssl_phase == "finetune":
            metrics_test = {
                k: pl_module.metrics[k] for k in pl_module.metrics if "test" in k
            }
            if metrics_test:
                self.on_epoch_end(trainer, metrics_test)
