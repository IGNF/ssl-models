"""Logger module."""

from typing import Literal

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torch.nn.functional import one_hot
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import InterpolationMode

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

    def __init__(self, log_inputs: list[str], **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.log_inputs = log_inputs

    @staticmethod
    def to_numpy(x: Tensor) -> Tensor:
        """Move to CPU and convert to numpy."""
        x_npy = x.detach().cpu().numpy().astype(np.float32)
        return (
            x_npy[:3]
            if x_npy.shape[0] >= BANDS_RGB
            else x_npy.mean(axis=0, keepdims=True)
        )

    @staticmethod
    def should_log(trainer: Trainer, outputs: dict) -> bool:
        """Check whether images should be logged."""
        return (
            trainer.global_rank == 0
            # and outputs["rec"] is not None
            # and outputs["masked"] is not None
        )

    def log_pretrain_images(
        self, trainer: Trainer,
        outputs: Tensor,
        batch: Tensor,
        log_index: str,
        epoch: int,
    ) -> None:
        for name_mod in outputs["masked"]:
            if name_mod not in self.log_inputs:
                continue
            trainer.tb_logger.add_image(
                f"{log_index}_{name_mod}_input",
                ImageLogger.to_numpy(outputs["masked"][name_mod][0, 0]),
                epoch,
            )
            trainer.tb_logger.add_image(
                f"{log_index}_{name_mod}_pred",
                ImageLogger.to_numpy(outputs["rec"][name_mod][0, 0]),
                epoch,
            )
            trainer.tb_logger.add_image(
                f"{log_index}_{name_mod}_target",
                ImageLogger.to_numpy(batch[name_mod][0, 0]),
                epoch,
            )

    def log_probe_or_finetune_images(
        self, trainer: Trainer,
        outputs: Tensor,
        batch: Tensor,
        log_index: str,
        epoch: int,
    ) -> None:
        ref_input = outputs["pred"].pop("ref_input")[0]
        for name_target, pred in outputs["pred"].items():
            # input mask
            image = batch[ref_input][0, 0][:3]
            mask = one_hot(batch[name_target][0, 0].type(torch.int64))
            mask = mask.squeeze().permute(2, 0, 1).type(torch.bool)
            mask = resize(
                mask,
                image.shape[1:],
                interpolation=InterpolationMode.NEAREST
            )
            mask_overlay = draw_segmentation_masks(
                image,
                masks=mask,
                alpha=0.2,
            )
            trainer.tb_logger.add_image(
                    f"{trainer.ssl_phase}_{name_target}_{log_index}_input",
                    ImageLogger.to_numpy(mask_overlay),
                    epoch,
                )
            # pred mask
            image = batch[ref_input][0, 0][:3]
            mask = pred[0, 0].type(torch.bool)
            mask = resize(mask, image.shape[1:])
            mask_overlay = draw_segmentation_masks(
                image,
                masks=mask,
                alpha=0.5,
            )
            trainer.tb_logger.add_image(
                    f"{trainer.ssl_phase}_{name_target}_{log_index}_pred",
                    ImageLogger.to_numpy(mask_overlay),
                    epoch,
                )

    def on_batch_end(
        self,
        trainer: Trainer,
        outputs: dict[str, dict],
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
        log_every_n_batches = round(
            num_batches_per_epoch / trainer.logged_images_per_epoch,
        )
        if batch_idx % log_every_n_batches:
            return

        epoch = trainer.current_epoch
        prefix_idx = batch_idx // log_every_n_batches
        if trainer.ssl_phase == "pretrain":
            log_index = f"{trainer.ssl_phase}_{stage}/{prefix_idx}"
            self.log_pretrain_images(trainer, outputs, batch, log_index, epoch)
        else:
            log_index = f"{stage}/{prefix_idx}"
            self.log_probe_or_finetune_images(
                trainer,
                outputs,
                batch,
                log_index,
                epoch
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
