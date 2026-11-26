"""Model conf."""

from dataclasses import dataclass, field

from hydra_zen import store

from ssl_models.conf.augment import AugmentConfig, DINOv3AugmentConfig
from ssl_models.conf.loss import DINOv3LossConfig, MAELossConfig
from ssl_models.conf.mask import DINOv3MaskConfig, MAEMaskConfig


@dataclass
class ModelConfig:
    """Shared model config.

    The EMA model (if any) is updated at each step or epoch (`ema_interval`),
    with a momentum following a cosine from `ema_momentum` (default
    `1 - 1 / (0.2 * nb of updates)`) to `ema_momentum_end` (constant if None).
    """

    fusion_mode: str = "group"
    type_head: str = "attentive"
    use_date_enc: bool = True
    ema_interval: str = "epoch"
    ema_momentum: float | None = None
    ema_momentum_end: float | None = None


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


@store(group="model", name="mae_model")
@dataclass
class MAEConfig(ModelConfig):  # noqa: D101
    inter_depth: int = 3  # nb of blocks with all mods
    model_size: str = "tiny"
    use_ema: bool = True
    condition: bool = False
    max_condition: int | None = None
    # MAE backbone (see `ssl_models.ssl.mae.MAE`), defaults match the former
    # `vit_pytorch` implementation (except for the decoder MLP width)
    norm: str = "layernorm"
    ffn: str = "mlp"
    layerscale_init: float | None = None
    drop_path: float = 0.0
    qkv_bias: bool = False
    inter_norm: bool = True  # norm before the inter encoder
    trunc_normal_init: bool = False
    patch_norm: bool = True
    pos_enc: str = "abs"  # "abs" (sin-cos) or "rope"
    use_cls: bool = False
    num_registers: int = 0
    rope_base: float = 100.0
    rope_shift: float | None = None
    rope_jitter: float | None = None
    rope_rescale: float | None = None

    augment: AugmentConfig = field(default_factory=AugmentConfig)
    mask: MAEMaskConfig = field(default_factory=MAEMaskConfig)
    loss: MAELossConfig = field(default_factory=MAELossConfig)


@store(group="model", name="dino_model")
@dataclass
class DINOv3Config(ModelConfig):
    # backbone
    inter_depth: int = 3
    model_size: str = "small"
    norm: str = "layernorm"
    num_registers: int = 4
    layerscale_init: float | None = 1e-5
    drop_path: float = 0.3
    patch_norm: bool = False
    inter_norm: bool = False  # norm before the inter encoder
    pos_enc: str = "rope"  # "abs" (sin-cos) or "rope"
    rope_base: float = 100.0
    rope_shift: float | None = None
    rope_jitter: float | None = None
    rope_rescale: float | None = 2.0
    # teacher, always used (optimization: see the `dinov3_pretrain` optimizer config)
    ema_interval: str = "step"
    ema_momentum: float | None = 0.994
    ema_momentum_end: float | None = 1.0

    augment: DINOv3AugmentConfig = field(default_factory=DINOv3AugmentConfig)
    mask: DINOv3MaskConfig = field(default_factory=DINOv3MaskConfig)
    loss: DINOv3LossConfig = field(default_factory=DINOv3LossConfig)
