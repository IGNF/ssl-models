"""Baseline Lightning Module."""

from typing import Literal

import torch

from ssl_models.baselines.croma import CROMABaseline
from ssl_models.baselines.dinov2 import Dinov2Baseline
from ssl_models.baselines.dofa import DOFABaseline
from ssl_models.baselines.prithvi import PrithviBaseline
from ssl_models.baselines.satmae import SatMAEBaseline
from ssl_models.baselines.terramind import TerramindBaseline
from ssl_models.conf.augment import AugmentConfig
from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.train.lit_base import LitBase

torch.set_float32_matmul_precision(precision="medium")


class LitBaseline(LitBase):
    """Baseline module for fine-tuning."""

    def __init__(
        self,
        # shared args
        dataset: DatasetConfig,
        augment: AugmentConfig | None,
        model: Literal["dinov2", "dofa", "croma", "prithvi", "satmae", "terramind"],
        fusion_mode: Literal["shared", "monotemp", "mod", "late-croma", "inter-croma"],
        model_size: Literal["small", "base", "large"],
        type_head: Literal["linear", "attentive"] = "attentive",
        weight_source: str = "imagenat",
        pretrained_path: str | None = None,
        freeze: bool = False,
        use_ema: bool = False,
        keep_norm: bool = True,
        use_date_enc: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(dataset=dataset, augment=augment)

        model_args = {
            "dataset": dataset,
            "fusion_mode": fusion_mode,
            "backbone_size": model_size,
            "freeze": freeze,
            "pretrained_path": pretrained_path,
            "type_head": type_head,
            "use_date_enc": use_date_enc,
            "keep_norm": keep_norm,
        }
        match model:
            case "dinov2" | "dofa":
                if fusion_mode not in ("shared", "monotemp"):
                    msg = f"Fusion mode {fusion_mode} not supported with {model}."
                    raise ValueError(msg)
            case "croma":
                if fusion_mode not in ("late-croma", "inter-croma"):
                    msg = f"Fusion mode {fusion_mode} not supported with {model}."
                    raise ValueError(msg)
            case "satmae" | "prithvi":
                if fusion_mode != "mod":
                    msg = f"Fusion mode {fusion_mode} not supported with {model}."
                    raise ValueError(msg)
            case "terramind":
                if fusion_mode != "shared":
                    msg = f"Fusion mode {fusion_mode} not supported with {model}."
                    raise ValueError(msg)
            case _:
                msg = f"Invalid model name {model}. Not implemented"
                raise ValueError(msg)

        match model:
            case "dinov2":
                model_args = {**model_args, "weight_source": weight_source}
                self.model = Dinov2Baseline(**model_args)
            case "dofa" | "croma" | "satmae":
                model_dict = {
                    "dofa": DOFABaseline,
                    "croma": CROMABaseline,
                    "satmae": SatMAEBaseline,
                }
                self.model = model_dict[model](**model_args)
            case "prithvi":
                model_args = {**model_args, "version": kwargs.get("version", "v2")}
                self.model = PrithviBaseline(**model_args)
            case "terramind":
                model_args = {**model_args, "version": kwargs.get("version", "v2")}
                self.model = TerramindBaseline(**model_args)

        self.init_ema(use_ema=use_ema, ema_interval="epoch")

        self.save_hyperparameters(ignore=["dataset", "augment"])

    def configure_optimizers(self) -> dict:
        """Configure optimizer."""
        total_steps = self.trainer.estimated_stepping_batches
        total_batch_size = (
            self.trainer.train_dataloader.batch_size
            * self.trainer.accumulate_grad_batches
            * self.trainer.num_nodes
            * self.trainer.num_devices
            / 3.0  # remain iso with existing runs on Jean Zellou
        )

        lr = (
            # use sqrt scaling rule with Adam
            self.trainer.base_lr * total_batch_size**0.5
        )
        match self.trainer.lw_decay:
            case None:
                grouped_params = self.model.parameters()
                max_lr = lr
            case _:
                grouped_params = self.model.grouped_parameters(
                    lr,
                    rate_decay=self.trainer.lw_decay,
                )

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
            final_div_factor=self.trainer.final_factor / 1000.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": f"{self.trainer.ssl_phase}_AdamW_lr",
            },
        }
