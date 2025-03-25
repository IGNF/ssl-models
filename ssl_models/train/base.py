"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

from abc import ABC, abstractmethod
from typing import Literal

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from conf.datasets import DatasetsConfig
from ssl_models.ssl.mae import MAE


class BaseModule(LightningModule, ABC):
    """Lightning module skeleton."""

    def __init__(
        self,
        # shared args
        datasets: DatasetsConfig,
        model,
        loss
    ):
        super().__init__()
        self.datasets = datasets
        self.model = model
        self.loss = loss

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
                if not isinstance(self.model, MAE):
                    msg = "Layer-wise decay only implemented for MAE."
                    raise NotImplementedError(msg)

                encoder = next(iter(self.model.encoder.values()))
                num_layers = len(encoder.layers)

                grouped_params = []
                grouped_params.append(
                    {
                        "params": self.model.patch_embed.parameters(),
                        "lr": lr * self.trainer.lw_decay ** (num_layers + 1),
                        "name": "layer_0_embed",
                    },
                )
                for idx in range(num_layers):
                    grouped_params.append(
                        {
                            "params": [
                                param
                                for encoder in self.model.encoder.values()
                                for param in encoder.layers[idx].parameters()
                            ],
                            "lr": lr * self.trainer.lw_decay ** (num_layers - idx),
                            "name": f"layer_{idx+1}",
                        },
                    )
                grouped_params.append(
                    {
                        "params": self.model.heads.parameters(),
                        "lr": lr,
                        "name": f"layer_{num_layers+1}_head",
                    },
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
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": f"{self.trainer.ssl_phase}_AdamW_lr",
            },
        }

    @abstractmethod
    def probe_or_finetune_step(self):
        """Probing/finetuning step."""
        return

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
        if stage != "train":
            return
        super().log(
            name=f"{ssl_phase}_{name}/step_{stage}",
            value=value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
