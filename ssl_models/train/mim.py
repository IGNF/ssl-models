"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

import copy
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.datasets import DatasetsConfig
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
from ssl_models.train.base import BaseModule
from ssl_models.train.condition_metric import ConditionMetric

torch.set_float32_matmul_precision(precision="medium")


class MiMModule(BaseModule):
    """Lightning SSL module."""

    def __init__(  # noqa: PLR0913, PLR0915, C901, PLR0912
        self,
        # shared args
        datasets: DatasetsConfig,
        mask: MiMMaskConfig,
        fusion_mode: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        inter_depth: int,
        model: Literal["mae", "mixmae"],
        model_size: Literal["tiny", "small", "base", "medium", "large", "huge"],
        type_head: Literal["linear", "attentive"] = "attentive",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        loss: MiMLossConfig | None = None,
        condition: bool = False,
        max_condition: int | None = None,
        use_date_enc: bool = True,
        use_ema: bool = False,
        # mixmae args
        num_levels: Literal[3, 4] = 3,
        swin_version: Literal[1, 2] = 1,
        v2_attn: bool = True,
        swin_abs_enc: bool = False,
    ) -> None:
        super().__init__(datasets)
        self.datasets = datasets

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
        self.condition_mods = {}
        for dataset in datasets.datasets.values():
            for name_mod, mod in dataset.inputs.items():
                dim = mod.patch_size.mae**2 * (
                    mod.bands
                    if isinstance(mod.bands, int)
                    else sum(map(len, mod.bands))
                )
                if (
                    name_mod in self.condition_mods
                    and self.condition_mods[name_mod] != dim
                ):
                    msg = "Inconsistent modality dimensionalities in multi-dataset"
                    raise ValueError(msg)
                self.condition_mods[name_mod] = dim
        for name_mod, dim in self.condition_mods.items():
            self.metrics[f"condition_{name_mod}"] = ConditionMetric(
                dim=dim,
            )

        self.norm_bands = {}
        for name_dataset, dataset in datasets.datasets.items():
            self.norm_bands[name_dataset] = {
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
                for name_mod, mod in dataset.inputs.items()
            }

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

        self.save_hyperparameters(ignore=["datasets", "mask", "loss"])

    def configure_optimizers(self) -> dict:
        """Configure optimizer."""
        total_steps = self.trainer.estimated_stepping_batches
        total_batch_size = (
            self.trainer.batch_size
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
        name_dataset: str,
        dataset: DatasetConfig,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Visu of reconstructed images."""
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_mod in pixels_rec:
            if name_mod not in dataset.log_inputs:
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
            log_inputs[f"{ssl_phase}_{stage}/_{name_dataset}_{name_mod}_input"] = (
                inputs[0, 0]
            )
            log_preds[f"{ssl_phase}_{stage}/_{name_dataset}_{name_mod}_rec"] = preds[
                0,
                0,
            ]
            log_targets[f"{ssl_phase}_{stage}/_{name_dataset}_{name_mod}_target"] = (
                targets[0, 0]
            )

        return log_inputs, log_preds, log_targets

    def compute_loss_rec(
        self,
        batch: dict[str, Tensor],
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        name_dataset: str,
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Compute reconstruction loss."""
        losses_rec = []
        weights = []
        for name_mod in pixels_rec:
            B, D = batch[name_mod].shape[:2]  # noqa: N806
            H, W = (  # noqa: N806
                self.model.out_grid_size[name_dataset][name_mod],
                self.model.out_grid_size[name_dataset][name_mod],
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
                        self.norm_bands[name_dataset][name_mod],
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
        self.metrics[f"loss_rec_{name_dataset}_{stage}"].update(loss_rec)
        return loss_rec, weight

    def pretrain_step(
        self,
        batch: dict[str, dict[str, Tensor]],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Pretraining step."""
        losses_rec, weights = {}, {}
        logs_inputs, logs_preds, logs_targets = {}, {}, {}
        for name_dataset, dataset in self.datasets.datasets.items():
            if batch[name_dataset] is None:
                continue

            batch[name_dataset], pixels_rec, mask_rec, _, _ = self.model(
                batch[name_dataset],
                name_dataset,
                ssl_phase=self.trainer.ssl_phase,
            )

            loss_rec, weight = self.compute_loss_rec(
                batch[name_dataset],
                pixels_rec,
                mask_rec,
                name_dataset,
                stage=stage,
            )
            losses_rec[name_dataset] = loss_rec
            weights[name_dataset] = weight

            log_inputs, log_preds, log_targets = self.compute_logs_rec(
                batch[name_dataset],
                pixels_rec,
                mask_rec,
                name_dataset,
                dataset,
                ssl_phase=self.trainer.ssl_phase,
                stage=stage,
            )
            logs_inputs.update(log_inputs)
            logs_preds.update(log_preds)
            logs_targets.update(log_targets)

        loss_rec = sum(
            losses_rec[name_dataset] * weights[name_dataset]
            for name_dataset in losses_rec
        ) / sum(weights.values())
        return {
            "loss": loss_rec,
            "losses": losses_rec,
            "log_inputs": logs_inputs,
            "log_preds": logs_preds,
            "log_targets": logs_targets,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        if not self.condition:
            return

        for name_dataset, dataset in self.datasets.datasets.items():
            self.condition_dataset(name_dataset, dataset)

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

    def condition_dataset(self, name_dataset: str, dataset: DatasetConfig) -> None:
        sampler = DistributedSampler(
            self.trainer.datamodule.train_dataset[name_dataset],
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False,  # or True if you want shuffle
        )

        data_loader = DataLoader(
            self.trainer.datamodule.train_dataset[name_dataset],
            batch_size=self.trainer.datamodule.batch_size[name_dataset],
            num_workers=self.trainer.datamodule.num_workers,
            sampler=sampler,
            drop_last=False,
        )

        with torch.no_grad():
            for batch in data_loader:
                batch = self.trainer.strategy.batch_to_device(batch, self.device)  # noqa: PLW2901
                batch = self.model.resize_and_rescale(batch, name_dataset)  # noqa: PLW2901

                for name_mod in dataset.inputs:
                    H = self.model.out_grid_size[name_dataset][name_mod]  # noqa: N806
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
