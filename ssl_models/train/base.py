"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

from abc import ABC
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.datasets import DatasetsConfig
from ssl_models.layers.overlay import (
    create_overlay,
    onehot_cd_from_logits,
    onehot_pred_from_logits,
    onehot_target_from_batch,
)
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric

RGB_BANDS = 3


class BaseModule(LightningModule, ABC):
    """Lightning base module."""

    def __init__(self, datasets: DatasetsConfig) -> None:
        super().__init__()
        self.datasets = datasets

        self.metrics = nn.ModuleDict()
        self.loss_pred = {}

        for name_dataset, dataset in datasets.datasets.items():
            self._init_dataset(name_dataset, dataset)

    def _init_dataset(
        self,
        name_dataset: str,
        dataset: DatasetConfig,
    ) -> None:
        for name_target, target in dataset.targets.items():
            if name_target in self.loss_pred:
                msg = "Target is defined multiple times"
                raise ValueError(msg)

            match target.type_target:
                case "classif" | "segment":
                    self.loss_pred[name_target] = F.cross_entropy
                    metric_partial = partial(
                        MonoLabelMetric,
                        type_target=target.type_target,
                        num_classes=target.num_classes,
                    )
                case "change_detect":
                    self.loss_pred[name_target] = F.binary_cross_entropy_with_logits
                    metric_partial = partial(
                        MonoLabelMetric,
                        type_target=target.type_target,
                        num_classes=target.num_classes,
                    )
                case "multilabel_classif":
                    self.loss_pred[name_target] = F.binary_cross_entropy_with_logits
                    metric_partial = partial(
                        MultiLabelMetric,
                        num_labels=target.num_classes,
                    )
            for stage in ("train", "val", "test"):
                self.metrics[f"{name_target}_{stage}"] = metric_partial()

        for name_loss in ("loss_rec", "loss_pred"):
            for stage in ("train", "val", "test"):
                self.metrics[f"{name_loss}_{name_dataset}_{stage}"] = MeanMetric(
                    dist_sync_on_step=False,
                )

    def compute_logs_pred(
        self,
        batch: dict[str, Tensor],
        logits: dict[str, Tensor],
        name_dataset: str,
        dataset: DatasetConfig,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
    ]:
        """Visu of predicted rasters."""
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_target, target in dataset.targets.items():
            if target.type_target in ("segment", "change_detect"):
                overlay_img = batch[dataset.log_inputs[0]][
                    0,
                    0,
                    :RGB_BANDS,
                ]
                onehot_target = onehot_target_from_batch(
                    batch[name_target][0, 0, 0],
                    target.num_classes,
                    target.missing_val,
                )
                log_inputs[
                    f"{ssl_phase}_{name_dataset}_{name_target}_{stage}/_input"
                ] = overlay_img
                log_targets[
                    f"{ssl_phase}_{name_dataset}_{name_target}_{stage}/_target"
                ] = create_overlay(
                    overlay_img,
                    onehot_target,
                    target.num_classes,
                )
            match target.type_target:
                case "segment":
                    onehot_pred = onehot_pred_from_logits(
                        logits[name_target][0, 0],
                        target.num_classes,
                    )
                    log_preds[
                        f"{ssl_phase}_{name_dataset}_{name_target}_{stage}/_pred"
                    ] = create_overlay(
                        overlay_img,
                        onehot_pred,
                        target.num_classes,
                    )
                case "change_detect":
                    overlay_img = batch[dataset.log_inputs[1]][
                        0,
                        0,
                        :RGB_BANDS,
                    ]
                    onehot_pred = onehot_cd_from_logits(logits)
                    log_preds[
                        f"{ssl_phase}_{name_dataset}_{name_target}_{stage}/_pred"
                    ] = create_overlay(
                        overlay_img,
                        onehot_pred,
                        target.num_classes,
                    )
        return log_inputs, log_preds, log_targets

    def compute_loss_pred(
        self,
        batch: dict[str, Tensor],
        logits: dict[str, Tensor],
        name_dataset: str,
        dataset: DatasetConfig,
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Compute prediction loss."""
        loss_pred = 0
        weight = 0
        for name_target, target in dataset.targets.items():
            logits_target = logits[name_target]
            targets = batch[name_target]

            match target.type_target:
                case "segment":
                    logits_target = rearrange(logits_target, "b 1 c h w -> (b h w) c")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.long()
                case "change_detect":
                    logits_target = rearrange(logits_target, "b 1 1 h w -> (b h w)")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.float()
                case "multilabel_classif":
                    targets = targets.float()
                case "classif":
                    targets = targets.long()

            if targets.ndim > 1:
                inds = (targets != target.missing_val).all(dim=1)
            else:
                inds = targets != target.missing_val

            weight += batch[name_target].shape[0] * inds.float().mean()

            inds = inds.nonzero().squeeze(dim=1)
            if len(inds) == 0:
                continue

            logits_selected = torch.index_select(
                logits_target,
                dim=0,
                index=inds,
            )
            targets_selected = torch.index_select(
                targets,
                dim=0,
                index=inds,
            )
            loss_pred += self.loss_pred[name_target](
                logits_selected,
                targets_selected,
            )
            self.metrics[f"{name_target}_{stage}"].update(
                logits_selected,
                targets_selected,
            )
        if not isinstance(loss_pred, Tensor):
            loss_pred = 0 * list(logits.values()).pop().mean()

        self.metrics[f"loss_pred_{name_dataset}_{stage}"].update(loss_pred)
        return loss_pred, weight

    def probe_or_finetune_step(
        self,
        batch: dict[str, dict[str, Tensor]],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Probing/finetuning step."""
        if (
            self.trainer.ssl_phase == "finetune"
            and stage != "train"
            and self.ema_model is not None
        ):
            model = self.ema_model
        else:
            model = self.model

        losses_pred, weights = {}, {}
        logs_inputs, logs_preds, logs_targets = {}, {}, {}
        for name_dataset, dataset in self.datasets.datasets.items():
            if batch[name_dataset] is None:
                continue

            batch[name_dataset], _, _, _, logits = model(
                batch[name_dataset],
                name_dataset,
                ssl_phase=self.trainer.ssl_phase,
            )

            loss_pred, weight = self.compute_loss_pred(
                batch[name_dataset],
                logits,
                name_dataset,
                dataset,
                stage=stage,
            )
            losses_pred[name_dataset] = loss_pred
            weights[name_dataset] = weight

            log_inputs, log_preds, log_targets = self.compute_logs_pred(
                batch[name_dataset],
                logits,
                name_dataset,
                dataset,
                ssl_phase=self.trainer.ssl_phase,
                stage=stage,
            )
            logs_inputs.update(log_inputs)
            logs_preds.update(log_preds)
            logs_targets.update(log_targets)

        loss_pred = sum(
            losses_pred[name_dataset] * weights[name_dataset]
            for name_dataset in losses_pred
        ) / sum(weights.values())
        return {
            "loss": loss_pred,
            "losses": losses_pred,
            "log_inputs": logs_inputs,
            "log_preds": logs_preds,
            "log_targets": logs_targets,
        }

    def shared_step(
        self,
        batch: dict[str, dict],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Shared step for training/validation."""
        # resize and rescale for all models here
        batch = self.resize_and_rescale(batch)

        match self.trainer.ssl_phase:
            case "pretrain":
                return self.pretrain_step(batch, stage)
            case "probe" | "finetune":
                return self.probe_or_finetune_step(batch, stage)
            case _:
                msg = (
                    f"Invalid ssl phase {self.trainer.ssl_phase}."
                    " Expected 'pretrain' or 'probe' or 'finetune'"
                )
                raise ValueError(msg)

    def training_step(
        self,
        batch: dict[str, dict],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="train")

    def validation_step(
        self,
        batch: dict[str, dict],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="val")

    def test_step(
        self,
        batch: dict[str, dict],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="test")

    def on_train_epoch_start(self) -> None:
        self.trainer.train_dataloader.set_epoch(self.trainer.current_epoch)

    def on_train_epoch_end(self) -> None:
        if self.ema_model is not None:
            self.update_ema()

    def update_ema(self) -> None:
        """Update EMA model."""
        momentum = 1 - 1 / (self.trainer.max_epochs * 0.2)
        for param, param_ema in zip(
            self.model.parameters(),
            self.ema_model.parameters(),
        ):
            param_ema.data.mul_(momentum).add_((1.0 - momentum) * param.detach().data)

    def resize_and_rescale(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Resize and/or rescale input rasters based on dataset config."""
        for name_mod, mod in self.dataset.inputs.items():
            batch[name_mod] = F.interpolate(
                batch[name_mod].flatten(0, 1),
                size=(mod.image_size,) * 2,
                mode=self.interpolate,
            ).unflatten(0, (-1, mod.num_dates))
            if mod.rescale_elev:
                batch[name_mod][:, :, 1:] = 30 * (
                    batch[name_mod][:, :, :1] - batch[name_mod][:, :, 1:]
                )
        return batch
