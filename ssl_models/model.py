"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

import itertools

import torch
import torch.nn.functional as F  # noqa: N812
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric

from ssl_models.mae import mae_large, mae_medium, mae_small, mae_tiny
from ssl_models.mixmae import (
    mixmae_base,
    mixmae_huge,
    mixmae_large,
    mixmae_small,
    mixmae_tiny,
)
from ssl_models.odeon_data import OdeonData

torch.set_float32_matmul_precision(precision="medium")


class SSLModule(LightningModule):
    """SSL module."""

    def __init__(
        self,
        model: str,
        model_size: str,
        mask_ratio: float = 0.75,
        image_size: int = 256,
        patch_size: int = 16,
        unpool: bool = False,
        type_head: str = "linear",
        loss: str = "l2_norm",
        swin_version: int | None = 1,
        v2_attn: bool = True,
        num_levels: int = 3,
        nb_mix_images: int | None = None,
        swin_abs_enc: bool = False,
    ) -> None:
        super().__init__()

        match loss:
            case "l1":
                norm_pix_loss = False
                loss_fn = torch.abs
            case "l2":
                norm_pix_loss = False
                loss_fn = torch.square
            case "l1_norm":
                norm_pix_loss = True
                loss_fn = torch.abs
            case "l2_norm":
                norm_pix_loss = True
                loss_fn = torch.square
            case _:
                msg = f"Invalid loss {loss}."
                raise ValueError(msg)

        match model:
            case "mae":
                num_levels = 1
                fac_abs_enc = 1.0  # abs pos encodings are necessary
                out_patch_size = patch_size

                model_map = {
                    "tiny": mae_tiny,
                    "small": mae_small,
                    "medium": mae_medium,
                    "large": mae_large,
                }
            case "mixmae":
                if num_levels not in (3, 4):
                    msg = f"Invalid number of levels {num_levels}. Expected '3' or '4'."
                    raise ValueError(msg)
                if swin_version not in (1, 2):
                    msg = f"Invalid swin version {swin_version}. Expected '1' or '2'"
                    raise ValueError(msg)
                fac_abs_enc = 1.0 if swin_abs_enc else 0.0  # optional abs pos encodings
                out_patch_size = patch_size * 2 ** (num_levels - 1)

                model_map = {
                    "tiny": mixmae_tiny,
                    "small": mixmae_small,
                    "base": mixmae_base,
                    "large": mixmae_large,
                    "huge": mixmae_huge,
                }
            case _:
                msg = f"Invalid model name {model}. Not implemented"
                raise ValueError(msg)

        if model_size in model_map:
            model_args = {
                "in_chans": 3,
                "mask_ratio": mask_ratio,
                "nb_mix_images": nb_mix_images,
                "image_size": image_size,
                "patch_size": patch_size,
                "out_patch_size": out_patch_size,
                "unpool": unpool,
                "num_levels": num_levels,
                "swin_version": swin_version,
                "v2_attn": v2_attn,
                "type_head": type_head,
                "loss_fn": loss_fn,
                "norm_pix_loss": norm_pix_loss,
                "fac_abs_enc": fac_abs_enc,
            }
            self.model = model_map[model_size](**model_args)
        else:
            msg = f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            raise ValueError(msg)

        self.num_classes = OdeonData.NUM_CLASSES
        self.missing_val = OdeonData.MISSING_VAL
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
        self.save_hyperparameters()

    def configure_optimizers(self) -> dict:
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
    ) -> dict:
        """Pretraining step."""
        loss, masked, pred, _ = self.model(
            batch["image"],
            ssl_phase=self.trainer.ssl_phase,
        )
        self.log(
            name="rec_loss",
            value=loss,
            ssl_phase=self.trainer.ssl_phase,
            phase=phase,
        )
        return {"loss": loss, "masked": masked, "pred": pred}

    def probe_or_finetune_step(
        self,
        batch: dict[str, torch.Tensor],
        phase: str,
    ) -> dict:
        """Probing/finetuning step."""
        _, _, _, head_logits = self.model(
            batch["image"],
            ssl_phase=self.trainer.ssl_phase,
        )
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

        self.log(
            name="cross_entropy_loss",
            value=loss,
            ssl_phase=self.trainer.ssl_phase,
            phase=phase,
        )
        return {"loss": loss, "masked": masked, "pred": pred}

    def shared_step(self, batch: dict[str, torch.Tensor], phase: str) -> dict:
        """Shared step for training/validation."""
        match self.trainer.ssl_phase:
            case "pretrain":
                return self.pretrain_step(batch, phase)
            case "probe" | "finetune":
                return self.probe_or_finetune_step(batch, phase)
            case _:
                msg = (
                    f"Invalid ssl phase {self.trainer.ssl_phase}."
                    " Expected 'pretrain' or 'probe' or 'finetune'"
                )
                raise ValueError(msg)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict:
        return self.shared_step(batch, phase="train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict:
        return self.shared_step(batch, phase="val")

    def on_epoch_end(self, phase: str) -> None:
        """Log epoch metrics for training/validation."""
        for dataset in self.num_classes:
            self.log_acc(
                name=f"acc_{dataset}",
                value=self.acc_metric[f"{dataset}_{phase}"].compute(),
                ssl_phase=self.trainer.ssl_phase,
                phase=phase,
            )
            self.acc_metric[f"{dataset}_{phase}"].reset()

    def on_train_epoch_end(self) -> None:
        if self.trainer.ssl_phase != "pretrain":
            self.on_epoch_end(phase="train")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.ssl_phase != "pretrain":
            self.on_epoch_end(phase="val")

    def log_acc(
        self,
        name: str,
        value: Tensor,
        ssl_phase: str,
        phase: str,
    ) -> None:
        """Log accuracy metric."""
        super().log(
            name=f"{ssl_phase}_{name}/{phase}",
            value=value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def log(self, name: str, value: Tensor, ssl_phase: str, phase: str) -> None:
        """Log metric."""
        for step_phase, on_step in zip([f"step_{phase}", f"{phase}"], [True, False]):
            if on_step and phase != "train":
                continue  # log by step only for train loss
            super().log(
                name=f"{ssl_phase}_{name}/{step_phase}",
                value=value,
                on_step=on_step,
                on_epoch=not on_step,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
