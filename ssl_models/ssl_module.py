"""
Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model
"""

import math

import torch
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy

from ssl_models.geopile_data import GeoPileDataset
from ssl_models.mae import mae_large, mae_medium, mae_small, mae_tiny
from ssl_models.mixmae import (
    mixmae_base,
    mixmae_huge,
    mixmae_large,
    mixmae_small,
    mixmae_tiny,
)

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
        type_probing="attentive",
    ):
        super().__init__()
        if type_probing is not None:
            self.num_classes = GeoPileDataset.num_classes
            self.missing_val = GeoPileDataset.missing_val
            self.cross_entropy = CrossEntropyLoss(ignore_index=self.missing_val)
            for dataset, num_classes in self.num_classes.items():
                acc_metric = MulticlassAccuracy(
                    num_classes=num_classes, ignore_index=self.missing_val
                )
                setattr(self, f"acc_metric_{dataset}", acc_metric)
        else:
            self.num_classes = {}
            self.missing_val = None
            self.cross_entropy = None
            self.acc_metric = None

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
                "type_probing": type_probing,
            }
            if band_groups:
                model_args["band_groups"] = band_groups
            if mask_batch:
                model_args["mask_batch"] = mask_batch
            self.model = model_map[model_size](**model_args)
        else:
            raise ValueError(
                f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            )
        self.tb_logger = None

    def forward(self, cube: dict[str, torch.Tensor]):
        return self.model(cube)

    def configure_optimizers(self):
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
                "monitor": "train/loss",
            },
        }

    def shared_step(self, batch: dict[str, torch.Tensor], batch_idx: int, phase: str):
        rec_loss, masked, pred, probe_logits = self(batch)
        probe_loss = 0
        for dataset in self.num_classes:
            if torch.all(batch[f"target_{dataset}"] == self.missing_val):
                continue
            probe_loss += self.cross_entropy(
                probe_logits[dataset], batch[f"target_{dataset}"]
            )
            acc_metric = getattr(self, f"acc_metric_{dataset}")
            probe_acc = acc_metric(probe_logits[dataset], batch[f"target_{dataset}"])
            self.log(name=f"{phase}/probe_acc_{dataset}", value=probe_acc)

        loss = rec_loss + probe_loss
        self.log(name=f"{phase}/rec_loss", value=rec_loss)
        self.log(name=f"{phase}/probe_loss", value=probe_loss)
        return {"loss": loss, "masked": masked, "pred": pred}

    def log(self, name, value):
        super().log(
            name=name,
            value=value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")
