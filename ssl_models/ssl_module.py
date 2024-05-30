"""
Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model
"""

import math

import torch
from pytorch_lightning import LightningModule

from ssl_models.mae import mae_large, mae_medium, mae_small, mae_tiny
from ssl_models.mixmae import mixmim_base, mixmim_large, mixmim_huge

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
        model_size="small",
        mask_ratio=0.75,
        image_size=512,
        patch_size=64,
        unpool=False,
        band_groups=None,
    ):
        super().__init__()
        # model_map = {
        #     "tiny": mae_tiny,
        #     "small": mae_small,
        #     "medium": mae_medium,
        #     "large": mae_large,
        # }
        model_map = {
            "base": mixmim_base,
            "large": mixmim_large,
            "huge": mixmim_huge,
        }
        num_downsampling = downsampling_from_patch_size(unpool, patch_size)
        if model_size in model_map:
            model_args = {
                "mask_ratio": mask_ratio,
                "image_size": image_size,
                "patch_size": patch_size,
                "num_downsampling": num_downsampling,
                "unpool": unpool,
            }
            if band_groups:
                model_args["band_groups"] = band_groups
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
        loss, masked, pred = self(batch)
        self.log(
            name=f"{phase}/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss, "masked": masked, "pred": pred}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")
