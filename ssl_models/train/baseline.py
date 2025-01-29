"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig
from ssl_models.ssl.mae import MAE
from ssl_models.baselines.dinov2 import DinoSegmentor
from ssl_models.baselines.dofa import DOFASegmentor
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
        in_channels: Literal["RGB", "RGBI", "IRC"] = "RGB",
        weight_source: str = 'imagenet',
        weights_path: str = None,

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
                
                ds_name = datasets.filter_finetune[0]
                ds = getattr(datasets, ds_name)
                ref_ds = getattr(ds, ds.ref_input)
                image_size = ref_ds.image_size

                model_args = {
                    "datasets": datasets,
                    "type_head": type_head,
                    "backbone_size": model_size,
                    "in_channels": in_channels,
                    "unpool_dim": unpool_dim,
                    "weight_source": weight_source,
                    "weights_path": weights_path,
                    "image_size": image_size
                }

                self.model = DinoSegmentor(**model_args)

            case "dofa":
                pass
            case _:
                msg = f"Invalid model name {model}. Not implemented"
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
        return {"loss": loss_pred, "masked": None, "rec": None}

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
            case "probe" | "finetune":
                return self.probe_or_finetune_step(batch, stage)
            case _:
                msg = (
                    f"Invalid ssl phase {self.trainer.ssl_phase}."
                    " Expected 'probe' or 'finetune'"
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

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="test")

    def on_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        """Log epoch metrics for training/validation."""
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
