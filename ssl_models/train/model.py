"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

import copy
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import v2

from ssl_models.conf.datasets import DatasetsConfig
from ssl_models.conf.mask import MaskConfig
from ssl_models.ssl.mae import mae_large, mae_medium, mae_small, mae_tiny
from ssl_models.ssl.mixmae import (
    mixmae_base,
    mixmae_huge,
    mixmae_large,
    mixmae_small,
    mixmae_tiny,
)
from ssl_models.train.base import BaseModule
from ssl_models import ROOT_DIR

torch.set_float32_matmul_precision(precision="medium")


class SSLModule(BaseModule):
    """Lightning SSL module."""

    def __init__(  # noqa: PLR0913
        self,
        # shared args
        datasets: DatasetsConfig,
        mask: MaskConfig,
        fusion_mode: Literal["shared", "monotemp", "mod", "group"],
        inter_depth: int,
        model: Literal["mae", "mixmae"],
        model_size: Literal["tiny", "small", "base", "medium", "large", "huge"],
        type_head: Literal["linear", "attentive"] = "attentive",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        loss: Literal["l1", "l2", "l1_norm", "l2_norm"] = "l2_norm",
        hkmeans_every_n_epochs: int | None = None,
        use_date_enc: bool = True,
        use_ema: bool = False,
        num_cls_tokens: int = 0,
        pool_classif: bool = True,
        decoder_depth: int = 3,
        decoder_dim: int = 512,
        decoder_heads: int = 16,
        distil_mod: str | None = None,
        distil_bands: list[int] = [0, 1, 2],
        distil_type: Literal["cls", "features"] = "cls",
        distil_lambda: float = 0.1,
        # mixmae args
        num_levels: Literal[3, 4] = 3,
        swin_version: Literal[1, 2] = 1,
        v2_attn: bool = True,
        swin_abs_enc: bool = False,
    ) -> None:
        super().__init__(datasets)

        self.distil_mod = distil_mod
        self.distil_bands = distil_bands
        self.distil_type = distil_type
        self.distil_lambda = distil_lambda
        if distil_mod:
            self.normalize = v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            self.dinov3_model = torch.hub.load(
                str(ROOT_DIR / "dinov3"),
                "dinov3_vitl16",
                source="local",
                weights=f"{str(ROOT_DIR)}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            )

        self.hkmeans_every_n_epochs = hkmeans_every_n_epochs

        self.norm_bands = {
            name_mod: tuple(
                (
                    mod.norm_bands
                    if mod.norm_bands is not None
                    else (
                        [mod.bands]
                        if isinstance(mod.bands, int)
                        else list(map(len, mod.bands))
                    )
                ),
            )
            for name_mod, mod in datasets.dataset.inputs.items()
        }
        match loss:
            case "l1":
                self.norm_pix_loss = False
                self.loss_fn = torch.abs
            case "l2":
                self.norm_pix_loss = False
                self.loss_fn = torch.square
            case "l1_norm":
                self.norm_pix_loss = True
                self.loss_fn = torch.abs
            case "l2_norm":
                self.norm_pix_loss = True
                self.loss_fn = torch.square
            case _:
                msg = f"Invalid loss {loss}."
                raise ValueError(msg)

        match model:
            case "mae":
                num_levels = 1
                fac_abs_enc = 1.0  # abs pos encodings are necessary for MAE
                fac_date_enc = 1.0 if use_date_enc else 0.0

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
                fac_date_enc = 1.0 if use_date_enc else 0.0

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

        if inter_depth and fusion_mode not in ("mod", "group"):
            msg = (
                "Intermediate fusion"
                f" not yet compatible with fusion mode {fusion_mode}."
            )
            raise NotImplementedError(msg)

        if model_size in model_map:
            model_args = {
                "datasets": datasets,
                "mask": mask,
                "fusion_mode": fusion_mode,
                "inter_depth": inter_depth,
                "model": model,
                "num_levels": num_levels,
                "swin_version": swin_version,
                "v2_attn": v2_attn,
                "type_head": type_head,
                "interpolate": interpolate,
                "fac_abs_enc": fac_abs_enc,
                "fac_date_enc": fac_date_enc,
                "num_cls_tokens": num_cls_tokens,
                "pool_classif": pool_classif,
                "decoder_depth": decoder_depth,
                "decoder_dim": decoder_dim,
                "decoder_heads": decoder_heads,
                "distil_mod": distil_mod,
            }
            self.model = model_map[model_size](**model_args)
        else:
            msg = f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            raise ValueError(msg)

        if use_ema:
            self.ema_model = copy.deepcopy(self.model).to("cpu")
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            self.ema_model = None

        self.dataset = datasets.dataset
        self.save_hyperparameters(ignore=["datasets", "mask"])

    def configure_optimizers(self) -> dict:
        """Configure optimizer."""
        total_steps = self.trainer.estimated_stepping_batches
        total_batch_size = (
            self.trainer.train_dataloader.batch_size
            * self.trainer.accumulate_grad_batches
            * self.trainer.num_nodes
            * self.trainer.num_devices
            / 3.0  # remain iso with past runs
        )
        lr = (
            # use sqrt scaling rule with Adam
            self.trainer.base_lr * total_batch_size**0.5
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

    def compute_logs_rec(
        self,
        batch: dict[str, Tensor],
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune", "balance"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Visu of reconstructed images."""
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_mod in pixels_rec:
            if name_mod not in self.dataset.log_inputs:
                continue
            inputs = torch.where(
                mask_rec[name_mod],
                0,
                batch[name_mod],
            )
            inputs = torch.where(
                torch.all(mask_rec[name_mod], dim=2, keepdim=True),
                1,
                inputs,
            )
            preds = torch.where(
                mask_rec[name_mod],
                pixels_rec[name_mod],
                batch[name_mod],
            )
            targets = batch[name_mod]
            log_inputs[f"{ssl_phase}_{stage}/_{name_mod}_input"] = inputs[0, 0]
            log_preds[f"{ssl_phase}_{stage}/_{name_mod}_rec"] = preds[0, 0]
            log_targets[f"{ssl_phase}_{stage}/_{name_mod}_target"] = targets[0, 0]

        return log_inputs, log_preds, log_targets

    def compute_loss_rec(
        self,
        batch: dict[str, Tensor],
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Compute reconstruction loss."""
        losses_rec = []
        weights = []
        for name_mod in pixels_rec:
            D = batch[name_mod].shape[1]  # noqa: N806
            H, W = (  # noqa: N806
                self.model.out_grid_size[name_mod],
                self.model.out_grid_size[name_mod],
            )
            P = batch[name_mod].shape[3] // H  # noqa: N806
            target = rearrange(
                batch[name_mod],
                "b d c (h p1) (w p2) -> b d (h w) (p1 p2) c",
                p1=P,
                p2=P,
            )
            if self.norm_pix_loss:
                target_groups = list(
                    torch.split(
                        target,
                        self.norm_bands[name_mod],
                        dim=-1,
                    ),
                )
                for idx, target_group in enumerate(target_groups):
                    mean = target_group.mean(dim=(-2, -1), keepdim=True)
                    var = target_group.var(dim=(-2, -1), keepdim=True)
                    target_groups[idx] = (target_group - mean) / (var + 1.0e-6) ** 0.5
                target = torch.cat(target_groups, dim=-1)
            target = rearrange(
                target,
                "b d (h w) (p1 p2) c -> b d c (h p1) (w p2)",
                h=H,
                p1=P,
                p2=P,
            )

            weight = D * H * W
            weights.append(weight)
            loss_rec = self.loss_fn(target - pixels_rec[name_mod])
            loss_rec = torch.masked_select(loss_rec, mask_rec[name_mod]).mean()
            losses_rec.append(weight * loss_rec)

        loss_rec = torch.stack(losses_rec).sum() / sum(weights)
        self.metrics[f"loss_rec_{stage}"].update(loss_rec)
        return loss_rec

    def pretrain_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Pretraining step."""
        batch, pixels_rec, mask_rec, _, _ = self.model(
            batch,
            ssl_phase=self.trainer.ssl_phase,
        )

        if self.distil_mod:
            _, _, _, x_distil, _ = self.model(batch, ssl_phase="distil")
            x_cls, x_distil = self.model.remove_cls_token(x_distil)

            inputs = self.dataset.inputs[self.distil_mod].norm_fac * F.interpolate(
                batch[self.distil_mod][:, 0, self.distil_bands],
                size=(self.model.out_grid_size[self.distil_mod] * 16,) * 2,
                mode="bilinear",
            )
            inputs = self.normalize(inputs / 255.0)
            with torch.no_grad():
                features = self.dinov3_model.forward_features_list([inputs], [None])[0]
                teacher = (
                    features["x_norm_clstoken"][:, None]
                    if self.distil_type == "cls"
                    else features["x_norm_patchtokens"]
                )

            student = (
                x_cls[self.distil_mod]
                if self.distil_type == "cls"
                else x_distil[self.distil_mod]
            )
            loss_distil = (
                1.0
                - F.cosine_similarity(
                    student.flatten(0, 1),
                    teacher.flatten(0, 1),
                ).mean()
            )
            self.metrics[f"loss_distil_{stage}"].update(loss_distil)

        loss_rec = self.compute_loss_rec(batch, pixels_rec, mask_rec, stage=stage)
        log_inputs, log_preds, log_targets = self.compute_logs_rec(
            batch,
            pixels_rec,
            mask_rec,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )

        return {
            "loss": loss_rec + self.distil_lambda * loss_distil,
            "log_inputs": log_inputs,
            "log_preds": log_preds,
            "log_targets": log_targets,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        if self.trainer.ssl_phase != "pretrain":
            self.trainer.datamodule.balance_weights = None

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        if self.hkmeans_every_n_epochs is None or self.trainer.ssl_phase != "pretrain":
            return

        dataset = self.trainer.datamodule.train_dataset
        if (self.trainer.current_epoch + 1) % self.hkmeans_every_n_epochs == 0:
            self.metrics["hkmeans"].reset()
            cls_tokens = []

            sampler = DistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=False,  # or True if you want shuffle
            )

            dl = self.trainer.datamodule.train_dataloader()
            dl = DataLoader(
                dataset,
                batch_size=dl.batch_size,
                sampler=sampler,
                num_workers=dl.num_workers,
                pin_memory=dl.pin_memory,
                drop_last=dl.drop_last,
                collate_fn=dl.collate_fn,
            )

            self.eval()
            with torch.no_grad():
                for batch in dl:
                    batch = self.trainer.strategy.batch_to_device(  # noqa: PLW2901
                        batch,
                        self.device,
                    )
                    cls_token = self.model(batch, ssl_phase="balance")
                    self.metrics["hkmeans"].update(ids=batch["idx"])
                    cls_tokens.append(cls_token.cpu())
            self.train()

            self.metrics["hkmeans"].compute_local(cls_tokens=cls_tokens)
            ids, weights = self.metrics["hkmeans"].compute()
            hkmeans_weights = torch.zeros(ids.max() + 1, device=self.device)
            hkmeans_weights[ids] += weights
            hkmeans_weights /= hkmeans_weights.mean()
            self.trainer.datamodule.balance_weights = hkmeans_weights.cpu().numpy()
