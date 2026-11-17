"""Utils to load configs."""

from pathlib import Path
from typing import Any

from hydra_zen import instantiate
from omegaconf import OmegaConf


def load_config(cfg_path: str | Path) -> dict[str, Any]:
    """Load the configs of a run from its `.hydra/config_resolved.yaml`.

    Returns a dict of instantiated configs (`run`, `dataset`, `model`, ...).
    """
    cfg = OmegaConf.load(cfg_path)
    return {key: instantiate(value) for key, value in cfg.items() if key != "_target_"}
