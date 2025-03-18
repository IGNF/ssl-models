"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""
import copy
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from conf.datasets import DatasetsConfig
from ssl_models.baselines.dinov2 import Dinov2Baseline
from ssl_models.baselines.dofa import DOFABaseline
from ssl_models.train.base import BaseModule

torch.set_float32_matmul_precision(precision="medium")

NDIM_RASTER = 5  # (batch, dates, channels, height, width)


class BaselineModule(BaseModule):
    """Baseline module for fine-tuning."""

    def __init__(
        self,
        # shared args
        datasets: DatasetsConfig,
        model: Literal["dinov2", "dofa"],
        model_size: Literal["small", "base", "large", "huge"],
        type_head: Literal["linear", "attentive"] = "linear",
        loss: Literal["l1", "l2", "l1_norm", "l2_norm"] = "l2_norm",
        weight_source: str = "imagenat",
        pretrained_path: str = None,
        freeze: bool = False,
        use_ema: bool = False,
        multimodal: str = "shared",
        keep_norm: bool = True,
    ) -> None:
        super().__init__(datasets, model, loss)

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
            case "dinov2":
                unpool_dim = None

                if len(datasets.filter_finetune)>1:
                    raise NotImplementedError("Too many datasets given.")
                
                model_args = {
                    "datasets": datasets,
                    "type_head": type_head,
                    "backbone_size": model_size,
                    "unpool_dim": unpool_dim,
                    "weight_source": weight_source,
                    "pretrained_path": pretrained_path,
                    "freeze": freeze,
                    "multimodal": multimodal,
                    "keep_norm": keep_norm
                }

                self.model = Dinov2Baseline(**model_args)

            case "dofa":
                unpool_dim = None

                if len(datasets.filter_finetune)>1:
                    raise NotImplementedError("Too many datasets given.")
                
                model_args = {
                    "datasets": datasets,
                    "type_head": type_head,
                    "backbone_size": model_size,
                    "unpool_dim": unpool_dim,
                    "freeze": freeze,
                    "pretrained_path": pretrained_path,
                    "multimodal": multimodal,
                    "keep_norm": keep_norm
                }

                self.model = DOFABaseline(**model_args)
            case _:
                msg = f"Invalid model name {model}. Not implemented"
                raise ValueError(msg)

        if use_ema:
            self.ema_model = copy.deepcopy(self.model).to("cpu")
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            self.ema_model = None

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
        match self.trainer.lw_decay:
            case None:
                grouped_params = self.model.parameters()
                max_lr = lr
            case _:
                grouped_params = self.model.grouped_parameters(lr, rate_decay=self.trainer.lw_decay)
                max_lr = [param["lr"] for param in grouped_params]

        optimizer = torch.optim.AdamW(
            grouped_params,
            lr=lr,
            weight_decay=self.trainer.wd,
            betas=(self.trainer.b1, self.trainer.b2),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
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

    def probe_or_finetune_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Probing/finetuning step."""
        if stage == "train" or self.ema_model is None:
            model = self.model
        else:
            model = self.ema_model

        _, log_inputs, log_preds, log_targets, loss_pred = model(
            batch,
            stage=stage,
            ssl_phase=self.trainer.ssl_phase,
        )

        return {
            "loss": loss_pred,
            "log_inputs": log_inputs,
            "log_preds": log_preds,
            "log_targets": log_targets,
        }

    def shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Shared step for training/validation."""
        # resize
        for name_mod, mod in self.dataset.inputs.items():
            batch[name_mod] = F.interpolate(
                batch[name_mod].flatten(0, 1),
                size=(mod.image_size,) * 2,
                mode="nearest",
            ).unflatten(0, (-1, mod.num_dates))
            batch[f"{name_mod}_dates"] = (
                F.interpolate(
                    batch[f"{name_mod}_dates"].float().flatten(0, 1),
                    size=(mod.image_size,) * 2,
                    mode="nearest",
                ).unflatten(0, (-1, mod.num_dates))
                if batch[f"{name_mod}_dates"].ndim == NDIM_RASTER
                else batch[f"{name_mod}_dates"].float()
            )

        match self.trainer.ssl_phase:
            case "finetune":
                return self.probe_or_finetune_step(batch, stage)
            case _:
                msg = (
                    f"Invalid ssl phase {self.trainer.ssl_phase}."
                    " Expected 'finetune'"
                )
                raise ValueError(msg)
            
    def on_train_epoch_end(self) -> None:
        if self.ema_model is not None:
            self.update_ema()

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

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="test")