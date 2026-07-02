"""Model conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class ModelConfig:  # noqa: D101
    fusion_mode: str = "group"
    inter_depth: int = 3  # nb of blocks with all mods
    model: str = "mae"
    model_size: str = "tiny"
    type_head: str = "attentive"
    interpolate: str = "bilinear"
    loss: str = "l1_norm"
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
    interpolate: str = "bilinear"
    freeze: bool = False
    weight_source: str = "imagenat"
    pretrained_path: str = None
    keep_norm: bool = True
    add_date_enc: bool = True
    use_ema: bool = True
    version: str = None
    # dinov3-specific args
    backbone_type: str = None
    repo_dir: str = "dinov3"
    hub_model: str = None
    checkpoint_path: str = None
    m2f_hidden_dim: int = 256
    interaction_indexes: list = None
    adapter_with_cp: bool = False


model_store = store(group="model")
model_store(ModelConfig, name="default_model")
model_store(BaselineConfig, name="baseline_model")

