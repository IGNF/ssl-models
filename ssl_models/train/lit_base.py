"""Base lightning module"""

import copy
from abc import ABC
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric

from ssl_models.conf.augment import AugmentConfig, DINOv3AugmentConfig
from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.layers.overlay import (
    create_overlay,
    onehot_cd_from_logits,
    onehot_pred_from_logits,
    onehot_target_from_batch,
)
from ssl_models.train.augment import Augment
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric
from ssl_models.train.schedule import cosine, lr_factor

RGB_BANDS = 3


class LitBase(LightningModule, ABC):
    """Lightning base module."""

    # random augmentations (crops, jitter, blur) also at val/test
    augment_eval: bool = False

    def __init__(
        self, dataset: DatasetConfig, augment: AugmentConfig | DINOv3AugmentConfig
    ) -> None:
        super().__init__()
        self.dataset = dataset

        # GPU augmentation (resize, flips, crops, jitter, blur), shared by all models
        self.augment = Augment(dataset, **vars(augment))

        # EMA of `self.model`, set with `init_ema` by subclasses that use it
        self.ema_model = None
        self.ema_interval = "epoch"
        self.ema_momentum = None
        self.ema_momentum_end = None
        self._last_ema_step = 0

        self.metrics = nn.ModuleDict()
        self.loss_pred = {}

        for name_target, target in self.dataset.targets.items():
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
                self.metrics[f"{name_loss}_{stage}"] = MeanMetric(
                    dist_sync_on_step=False,
                )

    def init_ema(
        self,
        use_ema: bool,
        ema_interval: Literal["epoch", "step"],
        ema_momentum: float | None,
        ema_momentum_end: float | None,
    ) -> None:
        """Create the EMA model as a frozen copy of `self.model` (see `ModelConfig`)."""
        if use_ema:
            self.ema_model = copy.deepcopy(self.model)
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.eval()
        self.ema_interval = ema_interval
        self.ema_momentum = ema_momentum
        self.ema_momentum_end = ema_momentum_end

    def total_batch_size(self) -> int:
        """Batch size per optimizer step, over accumulation and devices."""
        trainer = self.trainer
        return (
            trainer.opt.batch_size
            * trainer.accumulate_grad_batches
            * trainer.num_nodes
            * trainer.num_devices
        )

    def total_steps(self) -> int:
        """Total number of steps for a fit."""
        return max(int(self.trainer.estimated_stepping_batches), 1)

    def base_param_groups(self, lr: float) -> list[dict]:
        """Param groups before lr multipliers and weight decay exclusions."""
        return [{"params": self.model.parameters(), "lr": lr}]

    def param_groups(self, lr: float) -> list[dict]:
        """Optimizer param groups, with the opt config's lr multipliers and no wd.

        Each group of `base_param_groups` is split by lr multiplier (the
        `lr_mults` entry matching the param name, if any), weight decay exclusion
        (`no_wd_1d`, `no_wd_params`) and `freeze_params` match. Params without
        grad are left out. `wd_mult` (0 or 1) is used by the weight decay
        schedule to set the group's `weight_decay`, and `frozen` by the lr
        schedule to set the group's lr to 0 over the first `freeze_pct` steps.
        """
        opt = self.trainer.opt
        names = {id(param): name for name, param in self.model.named_parameters()}
        groups = {}
        for idx, base_group in enumerate(self.base_param_groups(lr)):
            for param in base_group["params"]:
                if not param.requires_grad:
                    continue
                name = names[id(param)]
                lr_mults = [mult for key, mult in opt.lr_mults.items() if key in name]
                if len(lr_mults) > 1:
                    msg = f"Parameter {name} matches several `lr_mults` keys."
                    raise ValueError(msg)
                lr_mult = lr_mults[0] if lr_mults else 1.0
                no_wd = (opt.no_wd_1d and param.ndim <= 1) or any(
                    key in name for key in opt.no_wd_params
                )
                frozen = any(key in name for key in opt.freeze_params)
                key = (idx, lr_mult, no_wd, frozen)
                if key not in groups:
                    wd_mult = 0.0 if no_wd else 1.0
                    groups[key] = {
                        "params": [],
                        "lr": base_group["lr"] * lr_mult,
                        "weight_decay": opt.wd * wd_mult,
                        "wd_mult": wd_mult,
                        "frozen": frozen,
                    }
                    if "name" in base_group:  # unique names for LearningRateMonitor
                        groups[key]["name"] = (
                            base_group["name"]
                            + (f"_lr_mult{lr_mult:g}" if lr_mult != 1 else "")
                            + ("_no_wd" if no_wd else "")
                            + ("_frozen" if frozen else "")
                        )
                groups[key]["params"].append(param)
        return list(groups.values())

    def configure_optimizers(self) -> dict:
        """AdamW with sqrt lr scaling, lr warmup + schedule (see `OptConfig`)."""
        opt = self.trainer.opt
        # sqrt scaling rule with Adam
        lr = opt.base_lr * (self.total_batch_size() / opt.lr_ref_batch_size) ** 0.5
        optimizer = torch.optim.AdamW(
            self.param_groups(lr),
            lr=lr,
            betas=(opt.b1, opt.b2),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                partial(
                    lr_factor,
                    total=self.total_steps(),
                    warmup_pct=opt.lr_warmup_pct,
                    schedule=opt.lr_schedule,
                    final_ratio=opt.final_lr_ratio,
                    freeze_pct=opt.freeze_pct if group["frozen"] else 0.0,
                )
                for group in optimizer.param_groups
            ],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": f"{self.trainer.ssl_phase}_AdamW_lr",
            },
        }

    def train(self, mode: bool = True) -> "LitBase":
        """Keep the EMA model in eval mode (no dropout / drop path)."""
        super().train(mode)
        if self.ema_model is not None:
            self.ema_model.eval()
        return self

    def on_train_epoch_start(self) -> None:
        self.trainer.datamodule.set_epoch(self.trainer.current_epoch)

    def on_train_batch_start(self, batch: dict, batch_idx: int) -> None:  # noqa: ARG002
        """Apply the weight decay schedule."""
        opt = self.trainer.opt
        wd_end = opt.wd if opt.wd_end is None else opt.wd_end
        weight_decay = cosine(opt.wd, wd_end, self.global_step, self.total_steps())

        for optimizer in self.trainer.optimizers:
            for group in optimizer.param_groups:
                group["weight_decay"] = weight_decay * group["wd_mult"]
        self.log(
            f"{self.trainer.ssl_phase}_AdamW_weight_decay", weight_decay, on_step=True
        )

    def on_train_batch_end(
        self,
        outputs: dict,  # noqa: ARG002
        batch: dict,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """EMA update after each optimizer step, if `ema_interval == "step"`."""
        if (
            self.ema_model is not None
            and self.ema_interval == "step"
            and self.global_step != self._last_ema_step  # skip accumulation steps
        ):
            self._last_ema_step = self.global_step
            self.update_ema()

    def on_train_epoch_end(self) -> None:
        """EMA update after each epoch, if `ema_interval == "epoch"`."""
        if self.ema_model is not None and self.ema_interval == "epoch":
            self.update_ema()

    def momentum(self) -> float:
        """EMA momentum of the current update, with a cosine schedule over the fit.

        From `ema_momentum` (default `1 - 1 / (0.2 * nb of updates)`) to
        `ema_momentum_end` (constant if None), with an update at each step or
        epoch (`ema_interval`). Computed at each fit: the number of updates
        differs between phases (pretrain / probe / finetune).
        """
        if self.ema_interval == "step":
            # called after the optimizer step, which increments `global_step`
            update, n_updates = self.global_step - 1, self.total_steps()
        else:
            update, n_updates = self.current_epoch, self.trainer.max_epochs
        start = self.ema_momentum
        if start is None:
            start = 1 - 1 / (0.2 * n_updates)
        end = start if self.ema_momentum_end is None else self.ema_momentum_end
        return cosine(start, end, update, n_updates)

    @torch.no_grad()
    def update_ema(self) -> None:
        """EMA update of `self.ema_model` from `self.model`, with `self.momentum()`."""
        momentum = self.momentum()
        self.log(f"{self.trainer.ssl_phase}_ema_momentum", momentum)
        params_ema = list(self.ema_model.parameters())
        params = list(self.model.parameters())
        torch._foreach_mul_(params_ema, momentum)  # noqa: SLF001
        torch._foreach_add_(params_ema, params, alpha=1.0 - momentum)  # noqa: SLF001

    def compute_logs_pred(
        self,
        batch: dict[str, Tensor],
        logits: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
    ]:
        """Visu of predicted rasters."""
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_target, target in self.dataset.targets.items():
            if target.type_target in ("segment", "change_detect"):
                overlay_img = batch[self.dataset.log_inputs[0]][
                    0,
                    0,
                    :RGB_BANDS,
                ]
                onehot_target = onehot_target_from_batch(
                    batch[name_target][0, 0, 0],
                    target.num_classes,
                    target.missing_val,
                )
                log_inputs[f"{ssl_phase}_{name_target}_{stage}/_input"] = overlay_img
                log_targets[f"{ssl_phase}_{name_target}_{stage}/_target"] = (
                    create_overlay(
                        overlay_img,
                        onehot_target,
                        target.num_classes,
                    )
                )
            match target.type_target:
                case "segment":
                    onehot_pred = onehot_pred_from_logits(
                        logits[name_target][0, 0],
                        target.num_classes,
                    )
                    log_preds[f"{ssl_phase}_{name_target}_{stage}/_pred"] = (
                        create_overlay(
                            overlay_img,
                            onehot_pred,
                            target.num_classes,
                        )
                    )
                case "change_detect":
                    overlay_img = batch[self.dataset.log_inputs[1]][
                        0,
                        0,
                        :RGB_BANDS,
                    ]
                    onehot_pred = onehot_cd_from_logits(logits)
                    log_preds[f"{ssl_phase}_{name_target}_{stage}/_pred"] = (
                        create_overlay(
                            overlay_img,
                            onehot_pred,
                            target.num_classes,
                        )
                    )
        return log_inputs, log_preds, log_targets

    def compute_loss_pred(
        self,
        batch: dict[str, Tensor],
        logits: dict[str, Tensor],
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Compute prediction loss."""
        loss_pred = 0
        for name_target, target in self.dataset.targets.items():
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

        self.metrics[f"loss_pred_{stage}"].update(loss_pred)
        return loss_pred

    def probe_or_finetune_step(
        self,
        batch: dict[str, Tensor],
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

        batch, _, _, _, logits = model(batch, ssl_phase=self.trainer.ssl_phase)
        loss_pred = self.compute_loss_pred(batch, logits, stage=stage)
        log_inputs, log_preds, log_targets = self.compute_logs_pred(
            batch,
            logits,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        return {
            "loss": loss_pred,
            "log_inputs": log_inputs,
            "log_preds": log_preds,
            "log_targets": log_targets,
        }

    def shared_step(
        self,
        batch: dict[str, Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Shared step for training/validation."""
        # augment (resize and rescale included) for all models here: global views
        # at the keys of the batch, local views (if any) at `{key}_local`
        batch = self.augment(batch, random=stage == "train" or self.augment_eval)

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
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="train")

    def validation_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="val")

    def test_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="test")
