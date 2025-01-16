"""Logger module."""

from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import cm
from matplotlib.colors import to_hex
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torchvision.utils import draw_segmentation_masks

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

    def add_confusion_matrix(self, tag: str, figure: Tensor, epoch: int) -> None:
        """Add confusion matrix to tensorboard logger."""
        self._experiment.add_figure(tag, figure, epoch)

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
            iteration_key = {
                k: outputs["masked"][k]
                for k in outputs["masked"]
                if k in self.log_inputs
            }
            input_key = 'outputs["masked"][name_it]'
            pred_key = outputs["rec"]
            target_key = batch
            log_title = f"{trainer.ssl_phase}_{stage}/{prefix_idx}_{{name_mod}}_{{img}}"

            def _create_img(
                msk: Tensor,
                inp: bool,
                **kwargs,  # noqa: ANN003, ARG001
            ) -> tuple[
                Tensor,
                str,
            ]:
                log_prefix = "input" if inp else "rec"
                return (
                    msk[0, 0][:BANDS_RGB],
                    log_prefix,
                )

        else:
            iteration_key = outputs["pred"]["pred_msk"]
            input_key = "batch[next((x for x in self.log_inputs if x in batch))]"
            pred_key = outputs["pred"]["pred_msk"]
            target_key = outputs["pred"]["input_msk"]
            log_title = f"{trainer.ssl_phase}_{{task}}_{stage}/{prefix_idx}_{{img}}"

            def _create_img(
                name_it: str,
                input_img: Tensor,
                msk: Tensor,
                colors: list[str],
                inp: bool,
                **kwargs,  # noqa: ANN003, ARG001
            ) -> tuple[Tensor, str]:
                input_img = input_img[0, 0][:BANDS_RGB]
                if "change_detect" in name_it:
                    input_img_name = "image_1" if inp else "image_2"
                    input_img = batch[input_img_name][0, 0][:BANDS_RGB]
                    log_prefix = input_img_name
                    return (
                        (
                            draw_segmentation_masks(input_img, msk, 0.4, colors),
                            log_prefix,
                        )
                        if not inp
                        else (input_img, log_prefix)
                    )
                log_prefix = "input" if inp else "pred"
                log_img = (
                    input_img
                    if inp
                    else draw_segmentation_masks(input_img, msk, 0.4, colors)
                )
                return (log_img, log_prefix)

        for name_it in iteration_key:
            input_img = eval(input_key, locals())  # noqa: S307
            pred = pred_key[name_it]
            target = target_key[name_it]
            num_classes_pred = pred.shape[0]
            num_classes_target = target.shape[0]
            colormap = list(
                map(
                    to_hex,
                    cm.get_cmap(
                        "plasma",
                        max(num_classes_pred, num_classes_target),
                    ).colors,
                ),
            )
            # input
            log_img, log_prefix = _create_img(
                name_it=name_it,
                input_img=input_img,
                msk=input_img,
                colors=None,
                inp=True,
            )
            log_title_input = log_title.format(
                task=name_it,
                name_mod=name_it,
                img=log_prefix,
            )
            trainer.tb_logger.add_image(
                log_title_input,
                ImageLogger.to_numpy(log_img),
                epoch,
            )
            # pred
            log_img, log_prefix = _create_img(
                name_it=name_it,
                input_img=input_img,
                msk=pred,
                colors=colormap[:num_classes_pred],
                inp=False,
            )
            log_title_pred = log_title.format(
                task=name_it,
                name_mod=name_it,
                img=log_prefix,
            )
            trainer.tb_logger.add_image(
                log_title_pred,
                ImageLogger.to_numpy(log_img),
                epoch,
            )
            # target
            log_img, _ = _create_img(
                name_it=name_it,
                input_img=input_img,
                msk=target,
                colors=colormap[:num_classes_target],
                inp=False,
            )
            log_title_target = log_title.format(
                task=name_it,
                name_mod=name_it,
                img="target",
            )
            trainer.tb_logger.add_image(
                log_title_target,
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
        pl_module: LightningModule,
        metrics: dict[str, dict[str, Tensor]]
    ) -> None:
        for name_target, metrics_target in metrics.items():
            if "confusion_matrix" in metrics_target:
                cm = metrics_target["confusion_matrix"].cpu().numpy()
                cm_df = pd.DataFrame(cm)
                cm_figure = sn.heatmap(cm_df, annot=True).get_figure()
                log_cm_title = f"confusion_matrix_{name_target}/confusion_matrix"
                trainer.tb_logger.add_confusion_matrix(
                    log_cm_title,
                    cm_figure,
                    pl_module.current_epoch
                )
                cm_norm_df = pd.DataFrame(
                    np.divide(
                        cm.astype(np.float32),
                        np.sum(cm, axis=1)[:, np.newaxis],
                        out=np.zeros_like(cm.astype(np.float32)),
                        where=(np.sum(cm, axis=1)[:, np.newaxis] != 0)
                    )
                )
                cm_norm_figure = sn.heatmap(cm_norm_df, annot=True).get_figure()
                log_cm_norm_title = f"confusion_matrix_{name_target}/confusion_matrix_normalize"
                trainer.tb_logger.add_confusion_matrix(
                    log_cm_norm_title,
                    cm_norm_figure,
                    pl_module.current_epoch
                )

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if pl_module.metrics and trainer.ssl_phase == "finetune":
            metrics_val = {
                    k: pl_module.metrics[k]
                    for k in pl_module.metrics
                    if "val" in k
            }
            if metrics_val:
                self.on_epoch_end(trainer, pl_module, metrics_val)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if pl_module.metrics and trainer.ssl_phase == "finetune":
            metrics_test = {
                    k: pl_module.metrics[k]
                    for k in pl_module.metrics
                    if "test" in k
            }
            if metrics_test:
                self.on_epoch_end(trainer, pl_module, metrics_test)
