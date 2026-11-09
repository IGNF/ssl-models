"""Model conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class MiMConfig:  # noqa: D101
    fusion_mode: str = "group"
    inter_depth: int = 3  # nb of blocks with all mods
    model: str = "mae"
    model_size: str = "tiny"
    type_head: str = "attentive"
    interpolate: str = "bilinear"
    swin_version: int = 1
    v2_attn: bool = True
    num_levels: int = 3
    swin_abs_enc: bool = False
    use_date_enc: bool = True
    use_ema: bool = True
    condition: bool = False
    max_condition: int | None = None


@dataclass
class BaselineConfig:  # noqa: D101
    fusion_mode: str = "shared"
    model: str = "dinov2"
    model_size: str = "small"
    type_head: str = "attentive"
    interpolate: str = "nearest"
    freeze: bool = False
    weight_source: str = "imagenat"
    pretrained_path: str = None
    keep_norm: bool = True
    add_date_enc: bool = True
    use_ema: bool = True
    version: str = None


@dataclass
class DinoConfig:
    """DINOv3-style pretraining config, see `ssl_models.train.dino.DINOModule`.

    Optimizer hyperparameters (base_lr, betas, weight decay schedule, gradient
    clipping) are set here: from opt_pretrain, only epochs, batch_size and
    accumulate_grad_batches are used. Probing / finetuning are not supported yet,
    set opt_probe.epochs=0 and opt_finetune.epochs=0. iBOT masking parameters
    are set in the mask config, loss and head parameters in the loss config,
    multi-crop parameters in the augment config.
    """

    # backbone
    fusion_mode: str = "group"
    inter_depth: int = 3
    model_size: str = "small"
    norm: str = "layernorm"
    ffn: str | None = None  # None: default of the model size
    mlp_ratio: float | None = None  # None: default of the model size
    n_registers: int = 4
    layerscale_init: float | None = 1e-5
    drop_path: float = 0.3
    patch_norm: bool = False
    rope_base: float = 100.0
    rope_shift: float | None = None
    rope_jitter: float | None = None
    rope_rescale: float | None = 2.0
    use_date_enc: bool = True
    interpolate: str = "bilinear"
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


model_store = store(group="model")
model_store(MiMConfig, name="mim_model")
model_store(BaselineConfig, name="baseline_model")
model_store(DinoConfig, name="dino_model")
