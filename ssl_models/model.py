"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model"""

import math

import torch
from pytorch_lightning import LightningModule
from torch import nn

from ssl_models.mae import mae_large, mae_medium, mae_small, mae_tiny
from ssl_models.metric import ClassifMetric

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
    def __init__(
        self,
        model="MAE",
        model_size="small",
        mask_ratio=0.75,
        image_size=512,
        patch_size=64,
        unpool=False,
        band_groups=None,
        type_head="attentive",
    ):
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
            raise ValueError(
                f"Invalid model size {model_size}. Expected one of {model_map.keys()}",
            )
        self.metric = nn.ModuleDict(
            [["classif_train", ClassifMetric()], ["classif_val", ClassifMetric()]],
        )
        self.tb_logger = None
        self.save_hyperparameters()

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        lr = (
            # use sqrt scaling rule with Adam
            self.trainer.base_lr
            * self.trainer.train_dataloader.batch_size**0.5
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
                "monitor": "train/loss",
                "name": f"{self.trainer.ssl_phase}_AdamW/lr",
            },
        }

    def pretrain_step(self, batch: dict[str, torch.Tensor], phase: str):
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
    ):
        _, _, _, head_logits = self.model(batch, ssl_phase=self.trainer.ssl_phase)

        masked, pred = None, None
        loss = self.metric[f"classif_{phase}"].compute_loss(head_logits, batch)
        self.metric[f"classif_{phase}"].update(head_logits, batch)

        self.log_classif(
            name="cross_entropy_loss",
            value=loss,
            ssl_phase=self.trainer.ssl_phase,
            phase=phase,
            on_step=True,
        )
        return {"loss": loss, "masked": masked, "pred": pred}

    def shared_step(self, batch: dict[str, torch.Tensor], phase: str):
        if self.trainer.ssl_phase == "pretrain":
            return self.pretrain_step(batch, phase)
        else:
            return self.probe_or_finetune_step(batch, phase)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, phase="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, phase="val")

    def on_epoch_end(self, phase) -> None:
        agg_loss, agg_acc = self.metric[f"classif_{phase}"].compute()

        self.log_classif(
            name="cross_entropy_loss",
            value=agg_loss,
            ssl_phase=self.trainer.ssl_phase,
            phase=phase,
            on_step=False,
        )
        for dataset, acc in agg_acc.items():
            self.log_classif(
                name=f"acc_{dataset}",
                value=acc,
                ssl_phase=self.trainer.ssl_phase,
                phase=phase,
                on_step=False,
            )

    def on_train_epoch_end(self) -> None:
        if self.trainer.ssl_phase != "pretrain":
            self.on_epoch_end(phase="train")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.ssl_phase != "pretrain":
            self.on_epoch_end(phase="val")

    def log_classif(self, name, value, ssl_phase, phase, on_step):
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

    def log(self, loss_name, value, ssl_phase, phase):
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
