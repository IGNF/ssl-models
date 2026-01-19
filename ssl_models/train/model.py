"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

import copy
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler

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

torch.set_float32_matmul_precision(precision="medium")


class SSLModule(BaseModule):
    """Lightning SSL module."""

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        # shared args
        datasets: DatasetsConfig,
        mask: MaskConfig,
        fusion_mode: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        inter_depth: int,
        model: Literal["mae", "mixmae"],
        model_size: Literal["tiny", "small", "base", "medium", "large", "huge"],
        type_head: Literal["linear", "attentive"] = "attentive",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        loss: Literal["l1", "l2", "l1_norm", "l2_norm", "l1_ls", "l2_ls"] = "l2_norm",
        balance_condition: bool = False,
        max_balance: float = 10,
        condition: bool = False,
        max_condition: int | None = None,
        balance_loss_every_n_epochs: int | None = None,
        balance_loss_pow: int = 1,
        use_date_enc: bool = True,
        use_ema: bool = False,
        # mixmae args
        num_levels: Literal[3, 4] = 3,
        swin_version: Literal[1, 2] = 1,
        v2_attn: bool = True,
        swin_abs_enc: bool = False,
    ) -> None:
        super().__init__(datasets)

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
                self.norm_pix_loss = None
                self.loss_fn = torch.abs
            case "l2":
                self.norm_pix_loss = None
                self.loss_fn = torch.square
            case "l1_norm":
                self.norm_pix_loss = "ln"
                self.loss_fn = torch.abs
            case "l2_norm":
                self.norm_pix_loss = "ln"
                self.loss_fn = torch.square
            case "l1_ls":
                self.norm_pix_loss = "ls"
                self.loss_fn = torch.abs
            case "l2_ls":
                self.norm_pix_loss = "ls"
                self.loss_fn = torch.square
            case _:
                msg = f"Invalid loss {loss}."
                raise ValueError(msg)

        self.balance_condition = balance_condition
        self.max_balance = max_balance

        self.condition = condition
        self.max_condition = max_condition
        self.mu, self.cov_inv_sqrt, self.proj = {}, {}, {}

        self.balance_loss_every_n_epochs = balance_loss_every_n_epochs
        self.balance_loss_pow = balance_loss_pow

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

        self.dataset = datasets.dataset
        self.save_hyperparameters(ignore=["datasets"])

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


    def target_norm(self, target: Tensor, name_mod: str) -> Tensor:
        """Apply target normalization."""
        H = self.model.out_grid_size[name_mod]  # noqa: N806
        P = target.shape[3] // H  # noqa: N806

        target = rearrange(
            target,
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
        return rearrange(
            target,
            "b d (h w) (p1 p2) c -> b d c (h p1) (w p2)",
            h=H,
            p1=P,
            p2=P,
        )

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
            log_preds[f"{ssl_phase}_{stage}/_{name_mod}_rec"] = preds[0, 0]
            log_targets[f"{ssl_phase}_{stage}/_{name_mod}_target"] = targets[0, 0]

        return log_inputs, log_preds, log_targets

    def compute_loss_rec(
        self,
        batch: dict[str, Tensor],
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        stage: Literal["train", "val", "test"] | None,
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
                    "(b d h w) (p1 p2 c) -> b d c (h p1) (w p2)",
                    d=D,
                    h=H,
                    w=W,
                    p1=P,
                    p2=P,
                )
            else:
                target = batch[name_mod]

            target = self.target_norm(target, name_mod=name_mod)

            weight = D * H * W
            weights.append(weight)
            loss_rec = self.loss_fn(target - pixels_rec[name_mod])
            loss_rec = torch.masked_select(loss_rec, mask_rec[name_mod]).mean()
            losses_rec.append(weight * loss_rec)

        loss_rec = torch.stack(losses_rec).sum() / sum(weights)
        if stage is not None:
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

        if self.trainer.ssl_phase != "pretrain":
            self.trainer.datamodule.balance_weights = None
            return

        dataset = self.trainer.datamodule.full_train_dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False,  # or True if you want shuffle
        )

        data_loader = self.trainer.datamodule.train_dataloader()
        data_loader = DataLoader(
            dataset,
            batch_size=data_loader.batch_size,
            sampler=sampler,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last,
            collate_fn=data_loader.collate_fn,
        )

        with torch.no_grad():
            for batch in data_loader:
                batch = self.trainer.strategy.batch_to_device(batch, self.device)  # noqa: PLW2901
                batch = self.model.resize_and_rescale(batch)  # noqa: PLW2901

                for name_mod in self.dataset.inputs:
                    target = self.target_norm(batch[name_mod], name_mod=name_mod)

                    H = self.model.out_grid_size[name_mod]  # noqa: N806
                    P = target.shape[3] // H  # noqa: N806

                    target = rearrange(
                        target,
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

        for name_mod in self.dataset.inputs:
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
            self.proj[name_mod] = torch.tensor(
                np.linalg.inv(L)[: self.max_condition, :], device=cov.device,
            ).float()

        if not self.balance_condition:
            return

        with torch.no_grad():
            for batch in data_loader:
                batch = self.trainer.strategy.batch_to_device(batch, self.device)  # noqa: PLW2901
                batch = self.model.resize_and_rescale(batch)  # noqa: PLW2901

                mu, gram = {}, {}
                for name_mod in self.dataset.inputs:
                    target = self.target_norm(batch[name_mod], name_mod=name_mod)

                    B = target.shape[0]  # noqa: N806
                    H = self.model.out_grid_size[name_mod]  # noqa: N806
                    P = target.shape[3] // H  # noqa: N806

                    target = rearrange(
                        target,
                        "b d c (h p1) (w p2) -> (b d h w) (p1 p2 c)",
                        p1=P,
                        p2=P,
                    )
                    target = torch.matmul(
                        self.proj[name_mod],
                        target.transpose(0, 1),
                    ).transpose(
                        0,
                        1,
                    )
                    target = rearrange(
                        target,
                        "(b dhw) c -> b dhw c",
                        b=B,
                    )
                    mu[name_mod] = target.mean(dim=1)
                    gram[name_mod] = (
                        torch.bmm(torch.transpose(target, 1, 2), target)
                        / target.shape[1]
                    )
                self.metrics["balance_condition"].update(
                    ids=batch["idx"],
                    mu=mu,
                    gram=gram,
                )

        self.metrics["balance_condition"].compute_local(max_balance=self.max_balance)
        ids, weights = self.metrics["balance_condition"].compute()
        balance_weights = torch.zeros(ids.max() + 1, device=self.device)
        balance_weights[ids] += weights
        balance_weights /= balance_weights.mean()
        self.trainer.datamodule.balance_weights = balance_weights.cpu().numpy()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        if self.trainer.ssl_phase != "pretrain" or self.balance_loss_every_n_epochs is None:
            return

        if (self.trainer.current_epoch + 1) % self.balance_loss_every_n_epochs != 0:
            return

        dataset = self.trainer.datamodule.full_train_dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False,  # or True if you want shuffle
        )

        data_loader = self.trainer.datamodule.train_dataloader()
        data_loader = DataLoader(
            dataset,
            batch_size=data_loader.batch_size,
            sampler=sampler,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            drop_last=False,
            collate_fn=data_loader.collate_fn,
        )

        self.metrics["balance_loss"].reset()

        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(data_loader):
                batch = self.trainer.strategy.batch_to_device(batch, self.device)  # noqa: PLW2901

                batch, pixels_rec, mask_rec, _, _ = self.model(
                    batch,
                    ssl_phase=self.trainer.ssl_phase,
                )

                losses = []
                ids = []
                for idx in range(len(batch["idx"])):
                    loss = self.compute_loss_rec(
                        batch={
                            name_mod: batch[name_mod][idx : idx + 1]
                            for name_mod in batch
                        },
                        pixels_rec={
                            name_mod: pixels_rec[name_mod][idx : idx + 1]
                            for name_mod in pixels_rec
                        },
                        mask_rec={
                            name_mod: mask_rec[name_mod][idx : idx + 1]
                            for name_mod in mask_rec
                        },
                        stage=None,
                    )
                    losses.append(loss)
                    ids.append(batch["idx"][idx])

                self.metrics["balance_loss"].update(
                    ids=ids,
                    losses=losses,
                )

            ids, weights = self.metrics["balance_loss"].compute()
            weights = weights**self.balance_loss_pow

            balance_weights = torch.zeros(ids.max() + 1, device=self.device)
            balance_weights[ids] += weights
            balance_weights /= balance_weights.mean()
            self.trainer.datamodule.balance_weights = balance_weights.cpu().numpy()

