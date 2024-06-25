"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

import itertools
from typing import Any

import math

import torch
import torch.nn.functional as F  # noqa: N812
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric

from ssl_models.geopile_data import GeoPileDataset
from ssl_models.mae import mae_large, mae_medium, mae_small, mae_tiny

torch.set_float32_matmul_precision(precision="medium")


def downsampling_from_patch_size(unpool, patch_size):
    if unpool:
        num_downsampling = round(math.log2(patch_size))
        if 2**num_downsampling != patch_size:
            raise ValueError("Patch size must be a power of 2")
    else:
        num_downsampling = None
    return num_downsampling


class SSLModule(LightningModule):
    """SSL module."""

    def __init__(
        self,
        model_size: str,
        mask_ratio: float,
        image_size: int,
        patch_size: int,
        band_groups: dict[str, list],
        type_head: str,
    ) -> None:
        super().__init__()

        model_map = {
            "tiny": mae_tiny,
            "small": mae_small,
            "medium": mae_medium,
            "large": mae_large,
        }
        if model_size in model_map:
            model_args = {
                "mask_ratio": mask_ratio,
                "image_size": image_size,
                "patch_size": patch_size,
                "type_head": type_head,
            }
            if band_groups:
                model_args["band_groups"] = band_groups
            if mask_batch:
                model_args["mask_batch"] = mask_batch
            self.model = model_map[model_size](**model_args)
        else:
            raise ValueError
        self.num_classes = GeoPileDataset.num_classes
        self.missing_val = GeoPileDataset.missing_val
        self.acc_metric = nn.ModuleDict(
            [
                [
                    f"{dataset}_{phase}",
                    MeanMetric(dist_sync_on_step=False, sync_on_compute=False),
                ]
                for dataset, phase in itertools.product(
                    self.num_classes.keys(),
                    ["train", "val"],
                )
            ],
        )
        self.tb_logger = None

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer."""
        total_steps = self.trainer.estimated_stepping_batches
        lr = (
            # use sqrt scaling rule with Adam
            self.trainer.base_lr * self.trainer.train_dataloader.batch_size**0.5
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.trainer.wd,
            betas=(self.trainer.b1, self.trainer.b2),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.2,
            cycle_momentum=False,
            div_factor=1000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": f"{self.trainer.ssl_phase}_AdamW_lr",
            },
        }

    def pretrain_step(
        self,
        batch: dict[str, torch.Tensor],
        phase: str,
    ) -> dict[str, Any]:
        """Pretraining step."""
        loss, masked, pred, _ = self.model(batch, ssl_phase=self.trainer.ssl_phase)
        self.log(
            loss_name="rec_loss",
            value=loss,
            ssl_phase=self.trainer.ssl_phase,
            phase=phase,
        )
        return {"loss": loss, "masked": masked, "pred": pred}

    def probe_or_finetune_step(
        self,
        batch: dict[str, torch.Tensor],
        phase: str,
    ) -> dict[str, Any]:
        """Probing/finetuning step."""
        _, _, _, head_logits = self.model(batch, ssl_phase=self.trainer.ssl_phase)

        masked, pred = None, None
        loss = 0
        for dataset in self.num_classes:
            if torch.all(batch[f"target_{dataset}"] == self.missing_val):
                continue
            inds = (batch[f"target_{dataset}"] != self.missing_val).nonzero().squeeze()
            logits = torch.index_select(head_logits[dataset], dim=0, index=inds)
            target = torch.index_select(
                batch[f"target_{dataset}"],
                dim=0,
                index=inds,
            )
            loss += F.cross_entropy(logits, target)

            acc = logits.argmax(dim=1) == target
            self.acc_metric[f"{dataset}_{phase}"].update(
                acc.mean(dtype=torch.float),
                weight=acc.numel(),
            )

        self.log_classif(
            name="cross_entropy_loss",
            value=loss,
            ssl_phase=self.trainer.ssl_phase,
            phase=phase,
            on_step=True,
        )
        return {"loss": loss, "masked": masked, "pred": pred}

    def shared_step(self, batch: dict[str, torch.Tensor], phase: str) -> dict[str, Any]:
        """Shared step for training/validation."""
        match self.trainer.ssl_phase:
            case "pretrain":
                return self.pretrain_step(batch, phase)
            case "probe" | "finetune":
                return self.probe_or_finetune_step(batch, phase)
            case _:
                raise ValueError

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        return self.shared_step(batch, phase="train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        return self.shared_step(batch, phase="val")

    def on_epoch_end(self, phase: str) -> None:
        """Log epoch metrics for training/validation."""
        for dataset in self.num_classes:
            self.log_classif(
                name=f"acc_{dataset}",
                value=self.acc_metric[f"{dataset}_{phase}"].compute(),
                ssl_phase=self.trainer.ssl_phase,
                phase=phase,
                on_step=False,
            )
            self.acc_metric[f"{dataset}_{phase}"].reset()

    def on_train_epoch_end(self) -> None:
        if self.trainer.ssl_phase != "pretrain":
            self.on_epoch_end(phase="train")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.ssl_phase != "pretrain":
            self.on_epoch_end(phase="val")

    def log_classif(
        self,
        name: str,
        value: Tensor,
        ssl_phase: str,
        phase: str,
        *,
        on_step: bool,
    ) -> None:
        """Log classif metrics."""
        if on_step and phase != "train":
            return
        step_phase = f"step_{phase}" if on_step else phase
        super().log(
            name=f"{ssl_phase}_{name}/{step_phase}",
            value=value,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def log(self, loss_name: str, value: Tensor, ssl_phase: str, phase: str) -> None:
        """Log loss."""
        for step_phase, on_step in zip([f"step_{phase}", f"{phase}"], [True, False]):
            if on_step and phase != "train":
                continue  # log by step only for train loss
            super().log(
                name=f"{ssl_phase}_{loss_name}/{step_phase}",
                value=value,
                on_step=on_step,
                on_epoch=not on_step,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
