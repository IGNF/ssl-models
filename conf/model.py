"""Model conf."""

from dataclasses import dataclass, field

from hydra_zen import store
from typing import List

@dataclass
class ModelConfig:  # noqa: D101
    multimodal: str = "group"
    allmods_depth: int = 3  # nb of blocks with all mods
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
    batch_repeats: int | None = None
    use_ema: bool = False

@dataclass
class BaselineConfig:  # noqa: D101
    multimodal: str = "shared"
    model: str = "dinov2" 
    model_size: str = "small"
    type_head: str = "linear"
    loss: str = "l2_norm"
    freeze: bool = False
    weight_source: str = "imagenat"
    pretrained_path: str = None
    keep_norm: bool = True


model_store = store(group="model")
model_store(ModelConfig, name="default_model")
model_store(BaselineConfig, name="baseline_model")
