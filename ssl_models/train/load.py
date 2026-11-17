"""Build or reload the Lightning module matching a model config."""

from pathlib import Path

from ssl_models.conf.augment import DinoAugmentConfig
from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.model import BaselineConfig, DINOv3Config, MiMConfig
from ssl_models.conf.utils import load_config
from ssl_models.train.base import BaseModule
from ssl_models.train.baseline import BaselineModule
from ssl_models.train.dino import DINOModule
from ssl_models.train.mim import MiMModule


def build_module(
    model: MiMConfig | DINOv3Config | BaselineConfig,
    dataset: DatasetConfig,
    augment: DinoAugmentConfig,
    ckpt_path: str | Path | None = None,
) -> BaseModule:
    """Build the Lightning module matching `model`, from scratch or from a checkpoint.

    When loading a checkpoint, hyperparameters come from the checkpoint, except the
    configs ignored by `save_hyperparameters` (dataset, mask, loss, augment), which
    are taken from the arguments.
    """
    match model:
        case MiMConfig():
            module_class = MiMModule
            kwargs = {}
        case DINOv3Config():
            module_class = DINOModule
            kwargs = {"augment": augment}
        case BaselineConfig():
            module_class = BaselineModule
            kwargs = {}
        case _:
            msg = f"Unknown model config {type(model).__name__}."
            raise TypeError(msg)

    if ckpt_path is None:
        return module_class(dataset=dataset, **kwargs, **vars(model))

    return module_class.load_from_checkpoint(
        ckpt_path,
        map_location=lambda storage, _: storage,
        strict=False,
        dataset=dataset,
        **kwargs,
    )


def load_module(ckpt_path: str | Path) -> BaseModule:
    """Reload a trained module from a checkpoint, using its run's resolved config.

    For MiM and baseline modules, the bare torch model is `load_module(path).model`.
    """
    ckpt_path = Path(ckpt_path)
    cfg = load_config(ckpt_path.parent.parent / ".hydra" / "config_resolved.yaml")
    return build_module(cfg["model"], cfg["dataset"], cfg["augment"], ckpt_path)
