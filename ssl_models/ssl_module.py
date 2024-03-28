from typing import Any, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from odeon.metrics.types import OdnMetric  # type: ignore
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from segmentation_models_pytorch import create_model  # type: ignore
from segmentation_models_pytorch.losses import FocalLoss  # type: ignore
from torch import Tensor
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    PolynomialLR,
    ReduceLROnPlateau,
)
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy as Accuracy,  # type: ignore[attr-defined]
)
from torchmetrics.classification import MulticlassF1Score as F1Score
from torchmetrics.classification import MulticlassPrecision as Precision
from torchmetrics.classification import MulticlassSpecificity as Specificity

__all__ = ["SSLModule"]


def get_lr_scheduler(optimizer, scheduler: str = "CosineAnnealingLR"):
    if scheduler == "ReduceLROnPlateau":
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, patience=20),
            "monitor": "val_loss",
        }

    elif scheduler == "ExponentialLR":
        lr_scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.95),
            "monitor": "val_loss",
        }

    elif scheduler == "CosineAnnealingLR":
        lr_scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=10),
            "monitor": "val_loss",
        }

    elif scheduler == "LinearLR+PolyLR":
        sch = ChainedScheduler(
            [
                LinearLR(
                    optimizer, start_factor=1e-6, end_factor=1.0, total_iters=1000
                ),
                PolynomialLR(optimizer, power=1.0, total_iters=39000),
            ]
        )
        lr_scheduler = {"scheduler": sch, "monitor": "val_loss", "interval": "step"}
    elif scheduler is None:
        lr_scheduler = None
    else:
        raise ValueError(f"LR Scheduler '{scheduler}' is unknown.")
    return lr_scheduler


def configure_loss(device: Any, loss: str, weight: Optional[List] = None) -> nn.Module:
    wt: Optional[Tensor] = None
    if weight is not None:
        wt = torch.Tensor(weight)
    if loss == "ce":
        return nn.CrossEntropyLoss(reduction="mean", ignore_index=0, weight=wt).to(
            device=device
        )
    elif loss == "focal":
        return FocalLoss("multiclass", normalized=True, ignore_index=0).to(
            device=device
        )
    else:
        raise ValueError(
            f"Loss type '{loss}' is not valid. "
            f"Currently, supports 'bce', or 'focal' loss."
        )


def configure_metrics(
    num_classes: int = 4,
    ignore_index: int = 0,
    average: Literal["none", "macro", "micro", "weighted"] = "macro",
) -> Tuple[OdnMetric, OdnMetric, OdnMetric]:
    train_metrics = MetricCollection(
        {
            "acc": Accuracy(
                num_classes=num_classes, ignore_index=ignore_index, average=average
            ),
            "spec": Specificity(
                num_classes=num_classes, ignore_index=ignore_index, average=average
            ),
            "pre": Precision(
                num_classes=num_classes, ignore_index=ignore_index, average=average
            ),
            "f1": F1Score(
                num_classes=num_classes, ignore_index=ignore_index, average=average
            ),
        },
        prefix="train_",
    )
    val_metrics = train_metrics.clone(prefix="val_")
    test_metrics = train_metrics.clone(prefix="test_")
    return train_metrics, val_metrics, test_metrics


class SSLModule(LightningModule):
    """Module for SSL pretraining."""

    def __init__(
        self,
        classes: int = 4,
        model: nn.Module | None = None,
        model_name: str = "unet",
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        encoder_weights: Any = "imagenet",
        loss: str = "ce",
        lr: float = 0.0001,
        scheduler: str = "ReduceLROnPlateau",
        optimizer: str = "adam",
        weight: Optional[List] = None,
        use_syncbn: bool = False,
        ignore_index: int = 0,
        average: Literal["none", "macro", "micro", "weighted"] = "none",
        **kwargs: Any,
    ):
        self.classes = classes
        super(SSLModule, self).__init__()
        self.model = (
            create_model(
                arch=model_name,
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=self.classes,
                **kwargs,
            )
            if model is None
            else model
        )
        self.loss: str = loss
        self.criterion: nn.Module = configure_loss(
            device=self.device, loss=self.loss, weight=weight
        )
        self.lr: float = lr
        self.optimizer: str = optimizer
        self.scheduler: str = scheduler
        self.train_metrics, self.val_metrics, self.test_metrics = configure_metrics(
            num_classes=self.classes, ignore_index=ignore_index, average=average
        )
        self.use_syncbn: bool = use_syncbn
        if self.use_syncbn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def step(self, sample, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        target = sample["mask"].squeeze()
        logit = self.model(sample["image"])
        pred = torch.softmax(logit, dim=1)
        return target, pred, logit

    def step_losses(self, logit, target) -> List[Tensor]:
        losses = self.criterion(logit, target)
        return losses

    def compute_metrics(self, pred: Tensor, target: Tensor, stage: str = "fit") -> None:
        if stage == "fit":
            self.train_metrics(pred, target)
        elif stage == "validate":
            self.val_metrics(pred, target)
        elif stage == "test":
            self.test_metrics(pred, target)
        else:
            raise NotImplementedError(f"can't compute metrics for stage {stage}")

    def log_metrics_at_epoch_end(
        self, stage: str = "fit", rank_zero_only: bool = True
    ) -> None:
        if stage == "fit":
            metrics = self.train_metrics
        elif stage == "validate":
            metrics = self.val_metrics
        elif stage == "test":
            metrics = self.test_metrics
        else:
            raise NotImplementedError(f"can't compute metrics for stage {stage}")
        metric_collection = metrics.compute()
        for metric_name, metric_value in metric_collection.items():
            self.log(
                name=f"{metric_name}/Avg",
                value=metric_value.mean(),
                rank_zero_only=rank_zero_only,
            )
        metrics.reset()

    def training_step(self, sample, *args, **kwargs) -> STEP_OUTPUT:
        target, pred, logit = self.step(sample, *args, **kwargs)
        losses = self.step_losses(logit=logit, target=target)
        for idx, loss in enumerate(losses):
            self.log(name=f"loss-{idx}", value=loss, rank_zero_only=True)
        self.compute_metrics(pred, target, stage="fit")
        return {"pred": pred, "target": target, "loss": losses[0]}

    def validation_step(self, sample, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        target, pred, logit = self.step(sample, *args, **kwargs)
        losses = self.step_losses(logit=logit, target=target)
        for idx, loss in enumerate(losses):
            self.log(name=f"loss-{idx}", value=loss, rank_zero_only=True)
        self.compute_metrics(pred, target, stage="validate")
        return {"pred": pred, "target": target}

    def test_step(self, sample, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        target, pred, logit = self.step(sample, *args, **kwargs)
        losses = self.step_losses(logit=logit, target=target)
        for idx, loss in enumerate(losses):
            self.log(name=f"loss-{idx}", value=loss, rank_zero_only=True)
        self.compute_metrics(pred, target, stage="test")
        return {"pred": pred, "target": target}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log_metrics_at_epoch_end(stage="fit")

    def validation_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        self.log_metrics_at_epoch_end(stage="validate")

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.log_metrics_at_epoch_end(stage="test")

    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler.
        Returns:
            a dict according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if self.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

        sch = get_lr_scheduler(optimizer, scheduler=self.scheduler)
        if sch is None:
            return {"optimizer": optimizer}
        else:
            return {"optimizer": optimizer, "lr_scheduler": sch}
