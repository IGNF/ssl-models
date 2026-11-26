"""Loss conf."""

from dataclasses import dataclass


@dataclass
class MAELossConfig:
    """MAE reconstruction loss, see `ssl_models.train.lit_mae.LitMAE`.

    `name` is one of "l1", "l2", "l1_norm", "l2_norm", "l1_ls", "l2_ls".
    """

    name: str = "l1_norm"


@dataclass
class DINOv3LossConfig:
    """DINO + iBOT + KoLeo losses and heads.

    See `ssl_models.train.lit_dino.LitDINOv3`.
    """

    # heads
    dino_prototypes: int = 65536
    ibot_prototypes: int = 65536
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_layers: int = 3
    # losses
    student_temp: float = 0.1
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_pct: float = 0.3
    w_dino: float = 1.0
    w_ibot: float = 1.0
    w_koleo: float = 0.1
