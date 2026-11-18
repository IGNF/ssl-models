"""Model conf."""

from dataclasses import dataclass, field

from hydra_zen import store

from ssl_models.conf.augment import AugmentConfig, DINOv3AugmentConfig
from ssl_models.conf.loss import DINOv3LossConfig, MiMLossConfig
from ssl_models.conf.mask import DINOv3MaskConfig, MiMMaskConfig


@dataclass
class ModelConfig:
    fusion_mode: str = "group"
    type_head: str = "attentive"
    use_date_enc: bool = True


@store(group="model", name="baseline_model")
@dataclass
class BaselineConfig(ModelConfig):  # noqa: D101
    model: str = "dinov2"
    model_size: str = "small"
    freeze: bool = False
    weight_source: str = "imagenat"
    pretrained_path: str = None
    keep_norm: bool = True
    use_ema: bool = True
    version: str = None

    augment: AugmentConfig = field(default_factory=AugmentConfig)


@store(group="model", name="mim_model")
@dataclass
class MiMConfig(ModelConfig):  # noqa: D101
    inter_depth: int = 3  # nb of blocks with all mods
    model: str = "mae"
    model_size: str = "tiny"
    swin_version: int = 1
    v2_attn: bool = True
    num_levels: int = 3
    swin_abs_enc: bool = False
    use_ema: bool = True
    condition: bool = False
    max_condition: int | None = None

    augment: AugmentConfig = field(default_factory=AugmentConfig)
    mask: MiMMaskConfig = field(default_factory=MiMMaskConfig)
    loss: MiMLossConfig = field(default_factory=MiMLossConfig)


@store(group="model", name="dino_model")
@dataclass
class DINOv3Config(ModelConfig):
    # backbone
    inter_depth: int = 3
    model_size: str = "small"
    norm: str = "layernorm"
    n_registers: int = 4
    layerscale_init: float | None = 1e-5
    drop_path: float = 0.3
    patch_norm: bool = False
    rope_base: float = 100.0
    rope_shift: float | None = None
    rope_jitter: float | None = None
    rope_rescale: float | None = 2.0
    # optimization
    base_lr: float = 2e-3  # scaled by sqrt(total batch size / 1024)
    b1: float = 0.9
    b2: float = 0.999
    lr_warmup_pct: float = 0.1
    lr_schedule: str = "cosine"
    final_lr_ratio: float = 1e-3
    wd_start: float = 0.04
    wd_end: float = 0.2
    momentum_start: float = 0.994
    patch_embed_lr_mult: float = 0.2
    freeze_last_layer_pct: float = 0.01
    clip_grad: float | None = 3.0

    augment: DINOv3AugmentConfig = field(default_factory=DINOv3AugmentConfig)
    mask: DINOv3MaskConfig = field(default_factory=DINOv3MaskConfig)
    loss: DINOv3LossConfig = field(default_factory=DINOv3LossConfig)
