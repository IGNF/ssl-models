"""
Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model
"""
import numpy as np

import torch
from pytorch_lightning import LightningModule

from ssl_models.mae import mae_large, mae_medium, mae_small, mae_tiny
from ssl_models import COUNT

torch.set_float32_matmul_precision(precision="medium")


class SSLModule(LightningModule):
    def __init__(  # noqa: PLR0913
        self,
        model_size="small",
        mask_ratio=0.75,
        image_size=512,
        patch_size=64,
        lr=1e-2,
        wd=0.0,
        b1=0.9,
        b2=0.999,
        band_groups=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
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
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 0.99, last_epoch=-1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def log_tb_images(self, images, preds, batch_idx) -> None:
        global COUNT
        COUNT += 1
        for img_idx, (image, pred) in enumerate(zip(images, preds)):
            print(torch.nn.functional.l1_loss(image, pred))
            self.tb_logger._experiment.add_image(
                f"image_{batch_idx}_{img_idx}",
                image.detach().cpu().numpy().astype(np.float32),
                COUNT,
            )
            self.tb_logger._experiment.add_image(
                f"pred_{batch_idx}_{img_idx}",
                pred.detach().cpu().numpy().astype(np.float32),
                COUNT,
            )

    def shared_step(self, batch: dict[str, torch.Tensor], batch_idx: int, phase: str):
        cube = batch
        loss, pred = self(cube)
        self.log(
            name=f"{phase}/loss",
            value=loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if batch_idx % 10 == 0:  # Log every 10 batches
            self.log_tb_images(cube["image"], pred, batch_idx)

        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")
