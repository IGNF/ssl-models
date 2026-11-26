"""Optimizer options conf."""

from dataclasses import dataclass, field

from hydra_zen import store


@dataclass
class OptConfig:
    """Shared optimizer config: AdamW with lr and weight decay schedules.

    The lr is `base_lr * sqrt(total batch size / lr_ref_batch_size)`, with a
    linear warmup over `lr_warmup_pct` of the steps, then a cosine decay to
    `final_lr_ratio` times the lr ("cosine") or a constant lr ("constant").
    The weight decay follows a cosine from `wd` to `wd_end` (constant if None).

    Parameters are matched by substring of their name in the lightning module's
    `model` (e.g. "patch_embed" matches "patch_embed.s2.proj.weight").
    """

    lr_ref_batch_size: float = 3.0  # remain iso with past runs
    lr_warmup_pct: float = 0.2
    lr_schedule: str = "cosine"
    final_lr_ratio: float = 1e-7
    lr_mults: dict[str, float] = field(default_factory=dict)  # lr multipliers
    lw_decay: float | None = None  # layer-wise lr decay (baselines only)
    b1: float = 0.9
    b2: float = 0.99
    wd: float = 0.01
    wd_end: float | None = None
    no_wd_1d: bool = False  # no weight decay on 1D params (biases, norms, scales)
    no_wd_params: list[str] = field(default_factory=list)  # no weight decay
    freeze_params: list[str] = field(default_factory=list)  # no update at start
    freeze_pct: float = 0.0  # proportion of steps with `freeze_params` frozen
    clip_grad: float | None = None  # max gradient norm
    accumulate_grad_batches: int = 1
    monitor: str | None = None
    patience: int | None = None


@store(group="opt_pretrain", name="default_pretrain")
@dataclass
class OptPretrainConfig(OptConfig):  # noqa: D101
    base_lr: float = 3e-5
    epochs: int = 20
    batch_size: int = 32


@store(group="opt_pretrain", name="dinov3_pretrain")
@dataclass
class OptDINOv3PretrainConfig(OptPretrainConfig):
    """DINOv3 pretraining optimizer config."""

    base_lr: float = 2e-3
    lr_ref_batch_size: float = 1024.0
    lr_warmup_pct: float = 0.1
    final_lr_ratio: float = 1e-3
    lr_mults: dict[str, float] = field(default_factory=lambda: {"patch_embed": 0.2})
    b1: float = 0.9
    b2: float = 0.999
    wd: float = 0.04
    wd_end: float | None = 0.2
    no_wd_1d: bool = True
    no_wd_params: list[str] = field(
        default_factory=lambda: ["cls_token", "register_tokens", "mask_token"],
    )
    freeze_params: list[str] = field(default_factory=lambda: ["prototypes"])
    freeze_pct: float = 0.01
    clip_grad: float | None = 3.0


@store(group="opt_probe", name="default_probe")
@dataclass
class OptProbeConfig(OptConfig):  # noqa: D101
    base_lr: float = 1e-5
    epochs: int = 10
    batch_size: int = 32


@store(group="opt_finetune", name="default_finetune")
@dataclass
class OptFinetuneConfig(OptConfig):
    """Finetuning optimizer config.

    Use:
        - monitor=treesat_mlc_thresh/weighted_f1_val for TreeSatAI-TS
        - monitor=pastis_seg/average_iou_val for PASTIS-HD
        - monitor=cosia/average_iou_val for FLAIR
    """

    base_lr: float = 1e-5
    epochs: int = 20
    batch_size: int = 32
    final_lr_ratio: float = 0.5
    monitor: str | None = None
    patience: int | None = 20
