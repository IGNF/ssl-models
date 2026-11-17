"""Lightning module for DINOv3-style pretraining (DINO + iBOT + KoLeo).

The student is trained on 2 masked global views and N local views, against an
EMA teacher that sees the unmasked global views. Optimizer hyperparameters
(lr, betas, weight decay schedule, gradient clipping) are DINO specific and
set here; the optimizer config only provides epochs, batch size and gradient
accumulation.
"""

from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from ssl_models.conf.augment import DinoAugmentConfig
from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.loss import DINOv3LossConfig
from ssl_models.conf.mask import DINOv3MaskConfig
from ssl_models.dino.augment import MultiCropAugment
from ssl_models.dino.dino import (
    dinov3_base,
    dinov3_huge_plus,
    dinov3_large,
    dinov3_small,
    dinov3_tiny,
)
from ssl_models.dino.head import DINOHead
from ssl_models.dino.loss import (
    dino_loss,
    ibot_loss,
    koleo_loss,
    sinkhorn_knopp,
    teacher_stats,
)
from ssl_models.dino.masking import ibot_masks, view_keep
from ssl_models.dino.schedule import cosine, linear_warmup, lr_factor
from ssl_models.train.lit_base import LitBase

NO_WD_PARAMS = ("cls_token", "register_tokens", "mask_token")


class LitDINOv3(LitBase):
    """Lightning DINOv3-style pretraining module."""

    def __init__(  # noqa: PLR0913
        self,
        dataset: DatasetConfig,
        mask: DINOv3MaskConfig,
        loss: DINOv3LossConfig,
        augment: DinoAugmentConfig,
        # backbone
        fusion_mode: Literal["shared", "monotemp", "mod", "group"] = "group",
        type_head: Literal["linear", "attentive"] = "attentive",  # noqa: ARG002, no probing yet
        inter_depth: int = 3,
        model_size: Literal["tiny", "small", "base", "large", "huge_plus"] = "small",
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        n_registers: int = 4,
        layerscale_init: float | None = 1e-5,
        drop_path: float = 0.3,
        patch_norm: bool = False,
        rope_base: float = 100.0,
        rope_shift: float | None = None,
        rope_jitter: float | None = None,
        rope_rescale: float | None = 2.0,
        use_date_enc: bool = True,
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
        # optimization
        base_lr: float = 2e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        lr_warmup_pct: float = 0.1,
        lr_schedule: Literal["cosine", "constant"] = "cosine",
        final_lr_ratio: float = 1e-3,
        wd_start: float = 0.04,
        wd_end: float = 0.2,
        momentum_start: float = 0.994,
        patch_embed_lr_mult: float = 0.2,
        freeze_last_layer_pct: float = 0.01,
        clip_grad: float | None = 3.0,
    ) -> None:
        super().__init__(dataset=dataset, interpolate=interpolate)

        model_map = {
            "tiny": dinov3_tiny,
            "small": dinov3_small,
            "base": dinov3_base,
            "large": dinov3_large,
            "huge_plus": dinov3_huge_plus,
        }
        if model_size not in model_map:
            msg = f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            raise ValueError(msg)

        model_args = {
            "dataset": dataset,
            "fusion_mode": fusion_mode,
            "inter_depth": inter_depth,
            "norm": norm,
            "n_registers": n_registers,
            "layerscale_init": layerscale_init,
            "drop_path": drop_path,
            "patch_norm": patch_norm,
            "rope_base": rope_base,
            "rope_shift": rope_shift,
            "rope_jitter": rope_jitter,
            "rope_rescale": rope_rescale,
            "local_ratio": augment.local_size_ratio,
            "fac_date_enc": 1.0 if use_date_enc else 0.0,
        }
        backbone = model_map[model_size](**model_args)

        head = partial(
            DINOHead,
            in_dim=backbone.embed_dim,
            hidden_dim=loss.head_hidden_dim,
            bottleneck_dim=loss.head_bottleneck_dim,
            n_layers=loss.head_n_layers,
        )
        # self.model is student, self.ema_model is teacher
        self.model = nn.ModuleDict(
            {
                "backbone": backbone,
                "dino_head": head(n_prototypes=loss.dino_n_prototypes),
                "ibot_head": head(n_prototypes=loss.ibot_n_prototypes),
            },
        )
        self.init_ema(use_ema=True, ema_interval="step")

        self.augment = MultiCropAugment(
            n_global=2,
            n_local=augment.n_local_crops,
            global_scale=tuple(float(v) for v in augment.global_crop_scale),
            local_scale=tuple(float(v) for v in augment.local_crop_scale),
            flip=augment.flip,
            jitter_prob=augment.jitter_prob,
            brightness=augment.brightness,
            contrast=augment.contrast,
            blur_min_size=augment.blur_min_size,
        )
        self.view_drop_mod = augment.view_drop_mod
        self.view_drop_dates = augment.view_drop_dates
        self.ibot_mask_prob = mask.ibot_mask_prob
        self.ibot_mask_ratio = tuple(float(v) for v in mask.ibot_mask_ratio)
        self.ibot_blob_grid = mask.ibot_blob_grid
        self.ibot_mask_shared = mask.ibot_mask_shared

        self.student_temp = loss.student_temp
        self.teacher_temp_start = loss.teacher_temp_start
        self.teacher_temp_end = loss.teacher_temp_end
        self.teacher_temp_warmup_pct = loss.teacher_temp_warmup_pct
        self.w_dino, self.w_ibot, self.w_koleo = loss.w_dino, loss.w_ibot, loss.w_koleo

        self.base_lr = base_lr
        self.betas = (b1, b2)
        self.lr_warmup_pct = lr_warmup_pct
        self.lr_schedule = lr_schedule
        self.final_lr_ratio = final_lr_ratio
        self.wd_start, self.wd_end = wd_start, wd_end
        self.momentum_start = momentum_start
        self.patch_embed_lr_mult = patch_embed_lr_mult
        self.freeze_last_layer_pct = freeze_last_layer_pct
        self.clip_grad = clip_grad

        self.total_steps = None

        self.save_hyperparameters(ignore=["dataset", "mask", "loss", "augment"])

    def get_total_steps(self) -> int:
        if self.total_steps is None:
            self.total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        return self.total_steps

    def teacher_temp(self) -> float:
        return linear_warmup(
            self.teacher_temp_start,
            self.teacher_temp_end,
            self.global_step,
            round(self.teacher_temp_warmup_pct * self.get_total_steps()),
        )

    def weight_decay(self) -> float:
        return cosine(
            self.wd_start, self.wd_end, self.global_step, self.get_total_steps()
        )

    def forward_batch(  # noqa: PLR0915
        self,
        batch: dict[str, Tensor],
        teacher_temp: float,
    ) -> dict:
        """Compute DINO / iBOT / KoLeo losses for a batch."""
        dataset = self.dataset
        backbone_s = self.model["backbone"]
        backbone_t = self.ema_model["backbone"]
        mods = list(dataset.inputs)

        B = batch[mods[0]].shape[0]  # noqa: N806
        device = batch[mods[0]].device
        views = self.augment(batch, backbone_s.view_size)
        n_global, n_local = len(views["global"]), len(views["local"])

        meta = {f"{m}_dates": batch[f"{m}_dates"] for m in mods}
        meta["ref_date"] = batch["ref_date"]

        def stack(view_list: list[dict[str, Tensor]]) -> dict[str, Tensor]:
            n = len(view_list)
            out = {m: torch.cat([v[m] for v in view_list]) for m in mods}
            out.update(
                {k: v.repeat(n, *(1,) * (v.ndim - 1)) for k, v in meta.items()},
            )
            return out

        num_dates = {m: dataset.inputs[m].num_dates for m in mods}
        globals_ = stack(views["global"])
        ibot_mask = ibot_masks(
            backbone_s.token_shapes("global"),
            n_global * B,
            self.ibot_mask_prob,
            self.ibot_mask_ratio,
            self.ibot_blob_grid,
            self.ibot_mask_shared,
            device,
        )
        keep_global = view_keep(
            num_dates,
            n_global * B,
            self.view_drop_mod,
            self.view_drop_dates,
            device,
        )

        # teacher: unmasked global views
        with torch.no_grad():
            out_t = backbone_t(globals_, "global")
            teacher_dino = sinkhorn_knopp(
                self.ema_model["dino_head"](out_t["cls"]),
                teacher_temp,
            )

        # student: masked global views, local views
        out_s = backbone_s(
            globals_,
            "global",
            ibot_mask=ibot_mask,
            keep=keep_global,
        )
        student_global = self.model["dino_head"](out_s["cls"])
        student_local = None
        if n_local:
            out_l = backbone_s(
                stack(views["local"]),
                "local",
                keep=view_keep(
                    num_dates,
                    n_local * B,
                    self.view_drop_mod,
                    self.view_drop_dates,
                    device,
                ),
            )
            student_local = self.model["dino_head"](out_l["cls"]).unflatten(
                0,
                (n_local, B),
            )

        losses_dino = dino_loss(
            student_global.unflatten(0, (n_global, B)),
            student_local,
            teacher_dino.unflatten(0, (n_global, B)),
            self.student_temp,
        )

        # iBOT on masked, not dropped, patch tokens of global views
        masked = torch.cat([ibot_mask[m].flatten(1, 2) for m in mods], dim=1)
        masked = masked & out_s["keep_flat"]
        num_masked = masked.sum(dim=1)
        with torch.no_grad():
            teacher_ibot = sinkhorn_knopp(
                self.ema_model["ibot_head"](out_t["patch_flat"][masked]),
                teacher_temp,
            )
        loss_ibot = ibot_loss(
            self.model["ibot_head"](out_s["patch_flat"][masked]),
            teacher_ibot,
            num_masked,
            self.student_temp,
        )

        cls_global = out_s["cls"].unflatten(0, (n_global, B))
        loss_koleo = torch.stack([koleo_loss(c) for c in cls_global]).mean()

        loss = (
            self.w_dino * losses_dino["total"]
            + self.w_ibot * loss_ibot
            + self.w_koleo * loss_koleo
        )
        stats = teacher_stats(teacher_dino)
        return {
            "loss": loss,
            "components": {
                "dino_global": losses_dino["global"].detach(),
                "dino_local": losses_dino["local"].detach(),
                "ibot": loss_ibot.detach(),
                "koleo": loss_koleo.detach(),
                "teacher_entropy": stats["entropy"],
                "teacher_kl_uniform": stats["kl_uniform"],
                "masked_ratio": masked.float().mean(),
            },
            "batch": batch,
            "views": views,
            "batch_size": B,
        }

    def pretrain_step(
        self,
        batch: dict[str, Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict:
        """Pretraining step."""
        ssl_phase = self.trainer.ssl_phase
        teacher_temp = self.teacher_temp()

        out = self.forward_batch(batch, teacher_temp)
        self.metrics[f"loss_rec_{stage}"].update(out["loss"].detach())
        for name, value in out["components"].items():
            self.log(
                f"{ssl_phase}_dino/{name}_{stage}",
                value,
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=out["batch_size"],
            )

        logs_inputs, logs_preds, logs_targets = {}, {}, {}
        for name_mod in self.dataset.log_inputs:
            prefix = f"{ssl_phase}_{stage}/_{name_mod}"
            logs_targets[f"{prefix}_original"] = out["batch"][name_mod][0, 0]
            logs_inputs[f"{prefix}_global"] = out["views"]["global"][0][name_mod][0, 0]
            if out["views"]["local"]:
                logs_preds[f"{prefix}_local"] = out["views"]["local"][0][name_mod][0, 0]

        if stage == "train":
            self.log(f"{ssl_phase}_dino/teacher_temp", teacher_temp, on_step=True)
            self.log(f"{ssl_phase}_dino/momentum", self.momentum, on_step=True)
            self.log(
                f"{ssl_phase}_dino/weight_decay", self.weight_decay(), on_step=True
            )
        return {
            "loss": out["loss"],
            "log_inputs": logs_inputs,
            "log_preds": logs_preds,
            "log_targets": logs_targets,
        }

    def shared_step(
        self,
        batch: dict[str, Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict:
        if self.trainer.ssl_phase != "pretrain":
            msg = "DINO module only supports pretraining for now."
            raise NotImplementedError(msg)
        return self.pretrain_step(batch, stage)

    def configure_optimizers(self) -> dict:
        """AdamW with no weight decay on biases/norms/tokens, warmup + schedule."""
        total_steps = self.get_total_steps()
        total_batch_size = (
            self.trainer.batch_size
            * self.trainer.accumulate_grad_batches
            * self.trainer.num_nodes
            * self.trainer.num_devices
        )
        lr = self.base_lr * (total_batch_size / 1024) ** 0.5

        groups = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            no_wd = param.ndim <= 1 or any(key in name for key in NO_WD_PARAMS)
            lr_mult = (
                self.patch_embed_lr_mult
                if name.startswith("backbone.patch_embed")
                else 1.0
            )
            key = (no_wd, lr_mult)
            if key not in groups:
                groups[key] = {
                    "params": [],
                    "lr": lr * lr_mult,
                    "wd_mult": 0.0 if no_wd else 1.0,
                    "weight_decay": 0.0 if no_wd else self.wd_start,
                }
            groups[key]["params"].append(param)

        optimizer = torch.optim.AdamW(list(groups.values()), lr=lr, betas=self.betas)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                lr_factor,
                total=total_steps,
                warmup=round(self.lr_warmup_pct * total_steps),
                schedule=self.lr_schedule,
                final_ratio=self.final_lr_ratio,
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": f"{self.trainer.ssl_phase}_AdamW_lr",
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip gradient norm to clip_grad unless set on the trainer."""
        clip_val = (
            gradient_clip_val if gradient_clip_val is not None else self.clip_grad
        )
        if clip_val:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    def on_train_batch_start(self, batch: dict, batch_idx: int) -> None:  # noqa: ARG002
        """Apply weight decay and teacher momentum schedules."""
        weight_decay = self.weight_decay()
        for optimizer in self.trainer.optimizers:
            for group in optimizer.param_groups:
                group["weight_decay"] = weight_decay * group.get("wd_mult", 1.0)
        # cosine schedule from `momentum_start` to 1, used by `update_ema`
        self.momentum = cosine(
            self.momentum_start, 1.0, self.global_step, self.get_total_steps()
        )

    def on_after_backward(self) -> None:
        """Freeze the prototypes at the beginning of training."""
        if self.global_step < round(
            self.freeze_last_layer_pct * self.get_total_steps()
        ):
            for name in ("dino_head", "ibot_head"):
                self.model[name].prototypes.grad = None
