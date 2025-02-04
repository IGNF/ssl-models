"""Model conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class ModelConfig:  # noqa: D101
    multimodal: str = "mod"
    model: str = "mae"
    model_size: str = "tiny"
    type_head: str = "attentive"
    loss: str = "l2_norm"
    unpool_dim: int | None = None
    swin_version: int = 1
    v2_attn: bool = True
    num_levels: int = 3
    swin_abs_enc: bool = False
    use_date_enc: bool = True

@dataclass
class BaselineConfig:  # noqa: D101
    model: str = "dinov2" 
    model_size: str = "small"
    type_head: str = "linear"
    loss: str = "l2_norm"
    in_channels: str = "RGB"
    freeze: bool = False
    weight_source: str = 'imagenet'
    weights_path: str = '/mnt/stores/store-DAI/equipiers/mvaccaro/foret/HighResCanopyHeight/compressed_SSLlarge.pth'

model_store = store(group="model")
model_store(ModelConfig, name="default_model")
model_store(BaselineConfig, name="baseline_model")
