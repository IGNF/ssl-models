"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

from typing import Literal

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ssl_models.conf.augment import AugmentConfig
from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.loss import MiMLossConfig
from ssl_models.conf.mask import MiMMaskConfig
from ssl_models.mim.mae import mae_large, mae_medium, mae_small, mae_tiny
from ssl_models.mim.mixmae import (
    mixmae_base,
    mixmae_huge,
    mixmae_large,
    mixmae_small,
    mixmae_tiny,
)
from ssl_models.train.condition_metric import ConditionMetric
from ssl_models.train.lit_base import LitBase

torch.set_float32_matmul_precision(precision="medium")


class LitMiM(LitBase):
    """Lightning SSL module."""

    def __init__(  # noqa: PLR0913, PLR0915, C901, PLR0912
        self,
        # shared args
        dataset: DatasetConfig,
        augment: AugmentConfig,
        mask: MiMMaskConfig,
        loss: MiMLossConfig,
        fusion_mode: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        inter_depth: int,
        model: Literal["mae", "mixmae"],
        model_size: Literal["tiny", "small", "base", "medium", "large", "huge"],
        type_head: Literal["linear", "attentive"] = "attentive",
        condition: bool = False,
        max_condition: int | None = None,
        use_date_enc: bool = True,
        use_ema: bool = False,
        ema_interval: Literal["epoch", "step"] = "epoch",
        ema_momentum: float | None = None,
        ema_momentum_end: float | None = None,
        # mae args
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        ffn: Literal["mlp", "swiglu"] = "mlp",
        layerscale_init: float | None = None,
        drop_path: float = 0.0,
        qkv_bias: bool = False,
        inter_norm: bool = True,
        trunc_normal_init: bool = False,
        # mixmae args
        num_levels: Literal[3, 4] = 3,
        swin_version: Literal[1, 2] = 1,
        v2_attn: bool = True,
        swin_abs_enc: bool = False,
    ) -> None:
        super().__init__(dataset=dataset, augment=augment)

        loss = loss or MiMLossConfig()
        match loss.name:
            case "l1":
                self.norm_pix_loss = None
                self.loss_fn = torch.abs
            case "l1_norm":
                self.norm_pix_loss = "ln"
                self.loss_fn = torch.abs
            case "l1_ls":
                self.norm_pix_loss = "ls"
                self.loss_fn = torch.abs
            case "l2":
                self.norm_pix_loss = None
                self.loss_fn = torch.square
            case "l2_norm":
                self.norm_pix_loss = "ln"
                self.loss_fn = torch.square
            case "l2_ls":
                self.norm_pix_loss = "ls"
                self.loss_fn = torch.square
            case _:
                msg = f"Invalid loss {loss.name}."
                raise ValueError(msg)

        self.condition = condition
        self.max_condition = max_condition

        self.mu = {}
        self.cov_inv_sqrt = {}
        self.condition_mods = {
            name_mod: mod.patch_size.mae**2
            * (mod.bands if isinstance(mod.bands, int) else sum(map(len, mod.bands)))
            for name_mod, mod in self.dataset.inputs.items()
        }
        for name_mod, dim in self.condition_mods.items():
            self.metrics[f"condition_{name_mod}"] = ConditionMetric(
                dim=dim,
            )

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
            for name_mod, mod in self.dataset.inputs.items()
        }

        match model:
            case "mae":
                num_levels = 1
                fac_abs_enc = 1.0  # abs pos encodings are necessary for MAE
                fac_date_enc = 1.0 if use_date_enc else 0.0
                backbone_args = {
                    "norm": norm,
                    "ffn": ffn,
                    "layerscale_init": layerscale_init,
                    "drop_path": drop_path,
                    "qkv_bias": qkv_bias,
                    "inter_norm": inter_norm,
                    "trunc_normal_init": trunc_normal_init,
                }

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
                backbone_args = {
                    "swin_version": swin_version,
                    "v2_attn": v2_attn,
                }

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
                "dataset": dataset,
                "mask": mask,
                "fusion_mode": fusion_mode,
                "inter_depth": inter_depth,
                "model": model,
                "num_levels": num_levels,
                "type_head": type_head,
                "fac_abs_enc": fac_abs_enc,
                "fac_date_enc": fac_date_enc,
                **backbone_args,
            }
            self.model = model_map[model_size](**model_args)
        else:
            msg = f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            raise ValueError(msg)

        self.init_ema(
            use_ema=use_ema,
            ema_interval=ema_interval,
            ema_momentum=ema_momentum,
            ema_momentum_end=ema_momentum_end,
        )

        self.save_hyperparameters(ignore=["dataset", "augment", "mask", "loss"])

    def compute_logs_rec(
        self,
        batch: dict[str, Tensor],
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
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
            log_preds[f"{ssl_phase}_{stage}/_{name_mod}_rec"] = preds[
                0,
                0,
            ]
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
            B, D = batch[name_mod].shape[:2]  # noqa: N806
            H, W = (  # noqa: N806
                self.model.out_grid_size[name_mod],
                self.model.out_grid_size[name_mod],
            )
            P = batch[name_mod].shape[3] // H  # noqa: N806

            if self.condition:
                target = rearrange(
                    batch[name_mod],
                    "b d c (h p1) (w p2) -> (b d h w) (p1 p2 c)",
                    p1=P,
                    p2=P,
                )
                target = target - self.mu[name_mod][None, :]
                target = torch.matmul(
                    self.cov_inv_sqrt[name_mod],
                    target.transpose(0, 1),
                ).transpose(
                    0,
                    1,
                )
                target = rearrange(
                    torch.tensor(target, device=batch[name_mod].device).float(),
                    "(b d h w) (p1 p2 c) -> b d (h w) (p1 p2) c",
                    d=D,
                    h=H,
                    w=W,
                    p1=P,
                    p2=P,
                )
            else:
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
                    if self.norm_pix_loss == "ln":
                        mean = target_group.mean(dim=(-2, -1), keepdim=True)
                        var = target_group.var(dim=(-2, -1), keepdim=True)
                        target_groups[idx] = (target_group - mean) / (
                            var + 1.0e-6
                        ) ** 0.5
                    else:
                        ms = torch.square(target_group).mean(dim=(-2, -1), keepdim=True)
                        target_groups[idx] = target_group / (ms + 1.0e-6) ** 0.5
                target = torch.cat(target_groups, dim=-1)
            target = rearrange(
                target,
                "b d (h w) (p1 p2) c -> b d c (h p1) (w p2)",
                h=H,
                p1=P,
                p2=P,
            )

            weight = B * D * H * W
            weights.append(weight)
            loss_rec = self.loss_fn(target - pixels_rec[name_mod])
            loss_rec = torch.masked_select(loss_rec, mask_rec[name_mod]).mean()
            losses_rec.append(weight * loss_rec)

        weight = sum(weights)
        loss_rec = torch.stack(losses_rec).sum() / weight
        self.metrics[f"loss_rec_{stage}"].update(loss_rec)
        return loss_rec

    def pretrain_step(
        self,
        batch: dict[str, Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Pretraining step."""
        batch, pixels_rec, mask_rec, _, _ = self.model(
            batch,
            ssl_phase=self.trainer.ssl_phase,
        )
        loss_rec = self.compute_loss_rec(batch, pixels_rec, mask_rec, stage=stage)
        log_inputs, log_preds, log_targets = self.compute_logs_rec(
            batch,
            pixels_rec,
            mask_rec,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        return {
            "loss": loss_rec,
            "log_inputs": log_inputs,
            "log_preds": log_preds,
            "log_targets": log_targets,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        if not self.condition:
            return

        self.condition_dataset()

        for name_mod in self.condition_mods:
            mu, gram = self.metrics[f"condition_{name_mod}"].compute()
            cov = gram - mu[:, None] * mu[None, :]
            V, L = np.linalg.eig(cov.cpu().numpy().astype(np.float64))  # noqa: N806
            if self.max_condition:
                cov_inv_sqrt = np.matmul(
                    L[:, : self.max_condition]
                    * np.sqrt(1 / np.maximum(V[None, : self.max_condition], 1e-10)),
                    np.linalg.inv(L)[: self.max_condition, :],
                )
            else:
                cov_inv_sqrt = np.matmul(L * np.sqrt(1 / V), np.linalg.inv(L))
            cov_inv_sqrt = torch.tensor(
                cov_inv_sqrt,
                device=cov.device,
            ).float()

            self.mu[name_mod] = mu
            self.cov_inv_sqrt[name_mod] = cov_inv_sqrt

    def condition_dataset(self) -> None:
        """Accumulate pixel statistics of the train dataset for conditioning."""
        sampler = DistributedSampler(
            self.trainer.datamodule.train_dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False,  # or True if you want shuffle
        )

        data_loader = DataLoader(
            self.trainer.datamodule.train_dataset,
            batch_size=self.trainer.datamodule.batch_size,
            num_workers=self.trainer.datamodule.num_workers,
            sampler=sampler,
            drop_last=False,
        )

        with torch.no_grad():
            for batch in data_loader:
                batch = self.trainer.strategy.batch_to_device(batch, self.device)  # noqa: PLW2901
                batch = self.augment(batch, random=False)  # noqa: PLW2901

                for name_mod in self.dataset.inputs:
                    H = self.model.out_grid_size[name_mod]  # noqa: N806
                    P = batch[name_mod].shape[3] // H  # noqa: N806

                    target = rearrange(
                        batch[name_mod],
                        "b d c (h p1) (w p2) -> (b d h w) (p1 p2 c)",
                        p1=P,
                        p2=P,
                    )
                    mu = target.mean(dim=0)
                    gram = (
                        torch.matmul(torch.transpose(target, 0, 1), target)
                        / target.shape[0]
                    )
                    count = target.shape[0]
                    self.metrics[f"condition_{name_mod}"].update(
                        mu=mu,
                        gram=gram,
                        count=count,
                    )
