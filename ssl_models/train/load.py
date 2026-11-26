"""Build or reload the Lightning module matching a model config."""

from pathlib import Path

from hydra_zen import instantiate
from omegaconf import OmegaConf

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.model import BaselineConfig, DINOv3Config, MAEConfig
from ssl_models.train.lit_base import LitBase
from ssl_models.train.lit_baseline import LitBaseline
from ssl_models.train.lit_dino import LitDINOv3
from ssl_models.train.lit_mae import LitMAE


def build_module(
    model: MAEConfig | DINOv3Config | BaselineConfig,
    dataset: DatasetConfig,
    ckpt_path: str | Path | None = None,
) -> LitBase:
    """Build the Lightning module matching `model`, from scratch or from a checkpoint.

    When loading a checkpoint, hyperparameters come from the checkpoint, except the
    configs ignored by `save_hyperparameters` (dataset, mask, loss, augment), which
    are taken from the arguments.
    """
    match model:
        case MAEConfig():
            module_class = LitMAE
            kwargs = {
                "dataset": dataset,
                "augment": model.augment,
                "mask": model.mask,
                "loss": model.loss,
            }
        case DINOv3Config():
            module_class = LitDINOv3
            kwargs = {
                "dataset": dataset,
                "augment": model.augment,
                "mask": model.mask,
                "loss": model.loss,
            }
        case BaselineConfig():
            module_class = LitBaseline
            kwargs = {
                "dataset": dataset,
                "augment": model.augment,
            }
        case _:
            msg = f"Unknown model config {type(model).__name__}."
            raise TypeError(msg)

    if ckpt_path is None:
        kwargs = {**vars(model), **kwargs}
        return module_class(**kwargs)

    return module_class.load_from_checkpoint(
        ckpt_path,
        map_location=lambda storage, _: storage,
        strict=False,
        **kwargs,
    )


def load_module(ckpt_path: str | Path) -> LitBase:
    """Reload a trained module from a checkpoint, using its run's resolved config.

    The bare torch model is `load_module(path).model` (and its EMA `.ema_model`,
    the teacher for DINOv3).
    """
    ckpt_path = Path(ckpt_path)
    cfg = OmegaConf.load(ckpt_path.parent.parent / ".hydra" / "config_resolved.yaml")
    model, dataset = instantiate(cfg.model), instantiate(cfg.dataset)
    return build_module(model, dataset, ckpt_path)
