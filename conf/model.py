"""Model conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class ModelConfig:  # noqa: D101
    multimodal: str = "group"
    allmods_depth: tuple[int] = (3, 3)  # nb of Transformer blocks with all mods
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


model_store = store(group="model")
model_store(ModelConfig, name="default_model")
