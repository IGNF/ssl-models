"""Model conf."""

from dataclasses import dataclass, field

from hydra_zen import builds, store

from ssl_models.model import SSLModule


@dataclass
class ModelConfig:  # noqa: D101
    model_size: str = "tiny"
    mask_ratio: float = 0.75
    image_size: int = 256
    patch_size: int = 16
    type_head: str = "attentive"
    band_groups: dict = field(default_factory=lambda: {"rgb": [0, 1, 2]})


model_build = builds(SSLModule, builds_bases=(ModelConfig,))

model_store = store(group="model")
model_store(model_build, name="default_model")
