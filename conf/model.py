"""Model conf."""

from dataclasses import dataclass

from hydra_zen import store

from conf import pbuilds
from ssl_models.ssl.model import SSLModule


@dataclass
class ModelConfig:  # noqa: D101
    model: str = "mae"
    model_size: str = "tiny"
    mask_ratio: float = 0.75
    image_size: int = 256
    patch_size: int = 16
    type_head: str = "linear"
    loss: str = "l2_norm"
    unpool: bool = False
    swin_version: int = 1
    v2_attn: bool = True
    num_levels: int = 3
    nb_mix_images: int = 4
    swin_abs_enc: bool = False


model_build = pbuilds(SSLModule, builds_bases=(ModelConfig,))

model_store = store(group="model")
model_store(model_build, name="default_model")
