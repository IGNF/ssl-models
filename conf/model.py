"""Model conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class ModelConfig:  # noqa: D101
    interpolate: str = "nearest"
    multimodal: str = "group"
    allmods_depth: int = 0  # nb of blocks with all mods
    model: str = "mae"
    model_size: str = "tiny"
    type_head: str = "attentive"
    loss: str = "l1_norm"
    unpool_dim: int | None = None
    swin_version: int = 1
    v2_attn: bool = True
    num_levels: int = 3
    swin_abs_enc: bool = False
    use_date_enc: bool = True
    use_ema: bool = True


@dataclass
class BaselineConfig:  # noqa: D101
    interpolate: str = "nearest"
    multimodal: str = "shared"
    model: str = "dinov2"
    model_size: str = "small"
    type_head: str = "attentive"
    freeze: bool = False
    weight_source: str = "imagenat"
    pretrained_path: str = None
    keep_norm: bool = True
    add_date_enc: bool = True
    use_ema: bool = True
    version: str = None


model_store = store(group="model")
model_store(ModelConfig, name="default_model")
model_store(BaselineConfig, name="baseline_model")
