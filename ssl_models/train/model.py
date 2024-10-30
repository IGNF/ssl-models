"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

from typing import Literal

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from conf.dataset.utils import InputConfig
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
                num_levels = 1
                fac_abs_enc = 1.0  # abs pos encodings are necessary

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
                "model": model,
                "mask_ratio": mask_ratio,
                "nb_mix_images": nb_mix_images,
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

        self.dataset = datasets.dataset
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
        loss_rec, masked, pred, _ = self.model(
            batch,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        self.log_step(
            name="loss_rec",
            value=loss_rec,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        return {"loss": loss_rec, "masked": masked, "pred": pred}

    def probe_or_finetune_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Probing/finetuning step."""
        _, _, _, loss_pred = self.model(
            batch,
            stage=stage,
            ssl_phase=self.trainer.ssl_phase,
        )

        self.log_step(
            name="loss_pred",
            value=loss_pred,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        return {"loss": loss_pred, "masked": None, "pred": None}

    def shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Shared step for training/validation."""
        for name_mod, mod in self.dataset.rasters.items():
            batch[name_mod] = F.interpolate(
                batch[name_mod].flatten(0, 1),
                size=(mod.image_size,) * 2,
                mode="bilinear" if isinstance(mod, InputConfig) else "nearest",
            ).unflatten(0, (-1, mod.num_dates))

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
        match self.trainer.ssl_phase:
            case "pretrain":
                self.log_metric(
                    name=f"{self.trainer.ssl_phase}_loss_rec/{stage}",
                    value=self.model.metrics[f"loss_rec_{stage}"].compute(),
                )
                self.model.metrics[f"loss_rec_{stage}"].reset()
            case "probe" | "finetune":
                self.log_metric(
                    name=f"{self.trainer.ssl_phase}_loss_pred/{stage}",
                    value=self.model.metrics[f"loss_pred_{stage}"].compute(),
                )
                self.model.metrics[f"loss_pred_{stage}"].reset()

                for name_target in self.dataset.targets:
                    metrics = self.model.metrics[f"{name_target}_{stage}"].compute()
                    for name_metric, value in metrics.items():
                        self.log_metric(
                            name=f"{self.trainer.ssl_phase}_{name_target}/{name_metric}_{stage}",
                            value=value,
                        )
                    self.model.metrics[f"{name_target}_{stage}"].reset()

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end(stage="train")

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end(stage="val")

    def log_metric(
        self,
        name: str,
        value: Tensor,
    ) -> None:
        """Log accuracy metric."""
        super().log(
            name=name,
            value=value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def log_step(
        self,
        name: str,
        value: Tensor,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> None:
        """Log metric."""
        super().log(
            name=f"{ssl_phase}_{name}/step_{stage}",
            value=value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
