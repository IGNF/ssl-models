"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

from typing import Literal

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from conf.datasets import DatasetsConfig
from ssl_models.ssl.mae import mae_large, mae_medium, mae_small, mae_tiny
from ssl_models.ssl.mixmae import (
    mixmae_base,
    mixmae_huge,
    mixmae_large,
    mixmae_small,
    mixmae_tiny,
)

torch.set_float32_matmul_precision(precision="medium")


class SSLModule(LightningModule):
    """SSL module."""

    def __init__(
        self,
        # shared args
        datasets: DatasetsConfig,
        model: Literal["mae", "mixmae"],
        model_size: Literal["tiny", "small", "base", "medium", "large", "huge"],
        type_head: Literal["linear", "attentive"] = "linear",
        loss: Literal["l1", "l2", "l1_norm", "l2_norm"] = "l2_norm",
        unpool: bool = False,
        # mae args
        mask_ratio: float = 0.75,
        # mixmae args
        nb_mix_images: int = 4,
        num_levels: Literal[3, 4] = 3,
        swin_version: Literal[1, 2] = 1,
        v2_attn: bool = True,
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
                image_size = datasets.dataset.image_size
                patch_size = datasets.mod.patch_size.mae

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
                image_size = datasets.mod.image_size
                patch_size = datasets.mod.patch_size.mixmae

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
                "datasets": datasets,
                "in_chans": datasets.mod.num_bands,
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

        self.mod_name = datasets.mod_name
        self.tb_logger = None
        self.save_hyperparameters(ignore=["datasets"])

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
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Pretraining step."""
        rec_loss, masked, pred, _ = self.model(
            batch,
            mod_name=self.mod_name,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        self.log(
            name="rec_loss",
            value=rec_loss,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        return {"loss": rec_loss, "masked": masked, "pred": pred}

    def probe_or_finetune_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Probing/finetuning step."""
        _, _, _, pred_loss = self.model(
            batch,
            mod_name=self.mod_name,
            stage=stage,
            ssl_phase=self.trainer.ssl_phase,
        )

        self.log(
            name="pred_loss",
            value=pred_loss,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        return {"loss": pred_loss, "masked": None, "pred": None}

    def shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Shared step for training/validation."""
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
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="val")

    def on_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        """Log epoch metrics for training/validation."""
        for name_target in self.model.targets:
            metrics = self.model.metrics[f"{name_target}_{stage}"].compute()
            for name_metric, value in metrics.items():
                self.log_acc(
                    name=f"{name_target}/{name_metric}",
                    value=value,
                    ssl_phase=self.trainer.ssl_phase,
                    stage=stage,
                )
            self.model.metrics[f"{name_target}_{stage}"].reset()

    def on_train_epoch_end(self) -> None:
        if self.trainer.ssl_phase != "pretrain":
            self.on_epoch_end(stage="train")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.ssl_phase != "pretrain":
            self.on_epoch_end(stage="val")

    def log_acc(
        self,
        name: str,
        value: Tensor,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> None:
        """Log accuracy metric."""
        super().log(
            name=f"{ssl_phase}_{name}/{stage}",
            value=value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def log(
        self,
        name: str,
        value: Tensor,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> None:
        """Log metric."""
        for step_stage, on_step in zip([f"step_{stage}", stage], [True, False]):
            if on_step and stage != "train":
                continue  # log by step only for train loss
            super().log(
                name=f"{ssl_phase}_{name}/{step_stage}",
                value=value,
                on_step=on_step,
                on_epoch=not on_step,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
