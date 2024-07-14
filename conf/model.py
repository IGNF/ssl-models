"""Model conf."""

from dataclasses import dataclass, field

from hydra_zen import builds, store

from ssl_models.model import SSLModule


@dataclass
class ModelConfig:  # noqa: D101
    model: str = "mae"
    model_size: str = "tiny"
    mask_ratio: float = 0.75
    image_size: int = 256
    patch_size: int = 16
    type_head: str = "linear"
    final_mlp: bool = False
    loss: str = "l2_norm"
    unpool: bool = False
    band_groups: dict = field(default_factory=lambda: {"rgb": [0, 1, 2]})
    swin_version: int = 2
    mask_batch: bool = False


model_build = builds(SSLModule, builds_bases=(ModelConfig,))

model_store = store(group="model")
model_store(model_build, name="default_model")
