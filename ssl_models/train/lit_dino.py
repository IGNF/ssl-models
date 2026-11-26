"""Lightning module for DINOv3-style pretraining (DINO + iBOT + KoLeo).

The student is trained on masked global views and local views, against an EMA
teacher that sees the unmasked global views. Views come from the shared
augmentation (`ssl_models.train.augment.Augment`, with `DINOv3AugmentConfig`),
with random crops, jitter and blur at train only (flips at all stages). The
DINOv3 module (`self.model`) samples the masks and applies the DINO and iBOT
heads; this module computes the losses from its outputs. Probing and finetuning
use its probing/finetuning heads, on a single global view.
The optimization is the shared one of `LitBase`, with the DINOv3 defaults of the
`dinov3_pretrain` optimizer config, and the teacher is the shared EMA model,
updated at each step; the teacher temperature schedule is set here. The DINO,
iBOT and KoLeo losses, with distributed Sinkhorn-Knopp centering, come from
`ssl_models.layers.loss`.
"""

from functools import partial
from typing import Literal

import torch
from torch import Tensor

from ssl_models.conf.augment import DINOv3AugmentConfig
from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.loss import DINOv3LossConfig
from ssl_models.conf.mask import DINOv3MaskConfig
from ssl_models.layers.loss import dino_loss, ibot_loss, koleo_loss, sinkhorn_knopp
from ssl_models.ssl.dino import (
    dinov3_base,
    dinov3_huge_plus,
    dinov3_large,
    dinov3_small,
    dinov3_tiny,
)
from ssl_models.train.lit_base import LitBase
from ssl_models.train.schedule import linear_warmup


@torch.no_grad()
def teacher_stats(probs: Tensor) -> dict[str, Tensor]:
    """Collapse diagnostics of teacher targets [N, K].

    "teacher_entropy": mean entropy of targets (0 = one-hot, log K = uniform),
    "teacher_kl_uniform": KL of the mean target to uniform (0 = all prototypes used).
    """
    probs = probs.float()
    K = probs.shape[1]  # noqa: N806
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1).mean()
    mean = probs.mean(dim=0)
    kl_uniform = (mean * (mean.clamp(min=1e-12) * K).log()).sum()
    return {"teacher_entropy": entropy, "teacher_kl_uniform": kl_uniform}


class LitDINOv3(LitBase):
    """Lightning DINOv3-style pretraining module."""

    def __init__(  # noqa: PLR0913
        self,
        dataset: DatasetConfig,
        augment: DINOv3AugmentConfig,
        mask: DINOv3MaskConfig,
        loss: DINOv3LossConfig,
        # backbone
        fusion_mode: Literal["shared", "monotemp", "mod", "group"] = "group",
        type_head: Literal["linear", "attentive"] = "attentive",
        inter_depth: int = 3,
        model_size: Literal["tiny", "small", "base", "large", "huge_plus"] = "small",
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        num_registers: int = 4,
        layerscale_init: float | None = 1e-5,
        drop_path: float = 0.3,
        patch_norm: bool = False,
        inter_norm: bool = False,
        pos_enc: Literal["abs", "rope"] = "rope",
        rope_base: float = 100.0,
        rope_shift: float | None = None,
        rope_jitter: float | None = None,
        rope_rescale: float | None = 2.0,
        use_date_enc: bool = True,
        # teacher
        ema_interval: Literal["epoch", "step"] = "step",
        ema_momentum: float | None = 0.994,
        ema_momentum_end: float | None = 1.0,
    ) -> None:
        super().__init__(dataset=dataset, augment=augment)
        if self.augment.num_global < 2 and not self.augment.num_local:  # noqa: PLR2004
            msg = (
                "DINO needs at least 2 views to compare, "
                f"got {self.augment.num_global} global and no local view."
            )
            raise ValueError(msg)

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
            "mask": mask,
            "fusion_mode": fusion_mode,
            "inter_depth": inter_depth,
            "norm": norm,
            "num_registers": num_registers,
            "layerscale_init": layerscale_init,
            "drop_path": drop_path,
            "patch_norm": patch_norm,
            "inter_norm": inter_norm,
            "type_head": type_head,
            "pos_enc": pos_enc,
            "rope_base": rope_base,
            "rope_shift": rope_shift,
            "rope_jitter": rope_jitter,
            "rope_rescale": rope_rescale,
            "local_ratio": self.augment.local_size_ratio,
            "fac_date_enc": 1.0 if use_date_enc else 0.0,
            "dino_prototypes": loss.dino_prototypes,
            "ibot_prototypes": loss.ibot_prototypes,
            "head_hidden_dim": loss.head_hidden_dim,
            "head_bottleneck_dim": loss.head_bottleneck_dim,
            "head_layers": loss.head_layers,
        }
        # self.model is student, self.ema_model is teacher
        self.model = model_map[model_size](**model_args)
        self.init_ema(
            use_ema=True,
            ema_interval=ema_interval,
            ema_momentum=ema_momentum,
            ema_momentum_end=ema_momentum_end,
        )

        self.student_temp = loss.student_temp
        self.teacher_temp_start = loss.teacher_temp_start
        self.teacher_temp_end = loss.teacher_temp_end
        self.teacher_temp_warmup_pct = loss.teacher_temp_warmup_pct
        self.teacher_temp = loss.teacher_temp_start  # set at each train batch start
        self.w_dino, self.w_ibot, self.w_koleo = loss.w_dino, loss.w_ibot, loss.w_koleo
        # losses on the views of the augmentation, stacked sample-major
        self.dino_loss = partial(
            dino_loss,
            num_global=self.augment.num_global,
            num_local=self.augment.num_local,
        )
        self.ibot_loss = ibot_loss
        self.koleo_loss = partial(koleo_loss, num_global=self.augment.num_global)

        self.save_hyperparameters(ignore=["dataset", "mask", "loss", "augment"])

    def on_train_batch_start(self, batch: dict, batch_idx: int) -> None:
        """Apply the parent schedules and, in pretraining, the teacher temperature."""
        super().on_train_batch_start(batch, batch_idx)

        if self.trainer.ssl_phase == "pretrain":
            self.teacher_temp = linear_warmup(
                self.teacher_temp_start,
                self.teacher_temp_end,
                self.global_step,
                round(self.teacher_temp_warmup_pct * self.total_steps()),
            )

    def compute_logs_dino(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Visu of the views: first sample (rows 0, 1, ...: its views), first date."""
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_mod in self.dataset.log_inputs:
            prefix = f"{ssl_phase}_{stage}/_{name_mod}"
            log_inputs[f"{prefix}_global"] = batch[name_mod][0, 0]
            if self.augment.num_global > 1:
                log_targets[f"{prefix}_global_2"] = batch[name_mod][1, 0]
            if self.augment.num_local:
                log_preds[f"{prefix}_local"] = batch[f"{name_mod}_local"][0, 0]
        return log_inputs, log_preds, log_targets

    def compute_loss_dino(
        self,
        student_global: dict[str, Tensor | None],
        student_local: dict[str, Tensor | None] | None,
        teacher_global: dict[str, Tensor | None],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, Tensor | float]:
        """Compute DINO / iBOT / KoLeo losses from the outputs of the student on
        the global (`student_global`) and local (`student_local`, None if no local
        view) views, and of the teacher on the global views (`teacher_global`),
        stacked sample-major ([B * n, ...]).

        Returns the total loss ("loss") and, to be logged, its components, the
        teacher temperature and the stats of the teacher targets.
        """
        teacher_dino = sinkhorn_knopp(teacher_global["dino"], self.teacher_temp)
        teacher_ibot = sinkhorn_knopp(teacher_global["ibot"], self.teacher_temp)

        loss_dino, loss_dino_global, loss_dino_local = self.dino_loss(
            student_global["dino"],
            student_local["dino"] if student_local is not None else None,
            teacher_dino,
            self.student_temp,
        )
        loss_ibot = self.ibot_loss(
            student_global["ibot"],
            student_global["ibot_weights"],
            teacher_ibot,
            self.student_temp,
        )
        loss_koleo = self.koleo_loss(student_global["cls"])

        loss = (
            self.w_dino * loss_dino
            + self.w_ibot * loss_ibot
            + self.w_koleo * loss_koleo
        )
        self.metrics[f"loss_rec_{stage}"].update(loss)

        losses = {
            "loss": loss,
            "loss_dino_global": loss_dino_global,
            "loss_dino_local": loss_dino_local,
            "loss_ibot": loss_ibot,
            "loss_koleo": loss_koleo,
            "teacher_temp": self.teacher_temp,
            **teacher_stats(teacher_dino),
        }
        return losses

    def pretrain_step(
        self,
        batch: dict[str, Tensor],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, Tensor]:
        """Pretraining step.

        Global views are at the keys of the batch, local views at `{key}_local`.
        """
        # student: masked global views, local views; teacher (frozen EMA copy, no
        # autograd graph): unmasked global views, with the iBOT head at the masked
        # positions of the student
        student_global = self.model(batch, ssl_phase="pretrain", view="global")
        student_local = (
            self.model(batch, ssl_phase="pretrain", view="local")
            if self.augment.num_local
            else None
        )
        teacher_global = self.ema_model(
            batch,
            ssl_phase="pretrain",
            view="global",
            teacher=True,
            mask_rec=student_global["mask_rec"],
        )

        losses = self.compute_loss_dino(
            student_global,
            student_local,
            teacher_global,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        log_inputs, log_preds, log_targets = self.compute_logs_dino(
            batch,
            ssl_phase=self.trainer.ssl_phase,
            stage=stage,
        )
        return {
            **losses,
            "log_inputs": log_inputs,
            "log_preds": log_preds,
            "log_targets": log_targets,
        }
