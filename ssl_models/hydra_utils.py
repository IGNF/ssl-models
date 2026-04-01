"""Hydra utils module."""

import uuid
from pathlib import Path

import re

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything


def pre_call_seed(cfg: OmegaConf) -> None:
    """Seed everything if run is reproducible."""
    if cfg.run.reproducible:
        seed_everything(cfg.run.seed, workers=True)


def get_epoch_safe(p):
    """Get epoch from checkpoint path."""
    m = re.search(r"epoch=(\d+)", str(p))
    return int(m.group(1)) if m else float("-inf")

def load_config(
    cfg: OmegaConf,
    load_name: str,
    load_phase: str,
) -> tuple[str, str]:
    """Load model config."""
    load_root = Path(cfg.run.exp_dir) / load_name
    ckpt_path = str(max(
        load_root.rglob(f"*/checkpoints/{load_phase}-epoch=*.ckpt"),
        key=get_epoch_safe,
    ))
    load_cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config_resolved.yaml"
    load_cfg = OmegaConf.load(load_cfg_path)

    return ckpt_path, load_cfg.run.exp_uuid


def pre_call_resolve(cfg: OmegaConf) -> None:
    """Resolve hydra config."""
    cfg.run.exp_uuid = uuid.uuid4()  # set uuid
    if cfg.run.load_name:
        cfg.run.load_ckpt_path, cfg.run.load_uuid = load_config(
            cfg,
            cfg.run.load_name,
            cfg.run.load_phase,
        )
    if cfg.run.fit_name:
        cfg.run.fit_ckpt_path, cfg.run.fit_uuid = load_config(
            cfg,
            cfg.run.fit_name,
            cfg.run.fit_phase,
        )

    hydra_dir = get_hydra_dir()
    cfg_path = hydra_dir / ".hydra" / "config_resolved.yaml"
    OmegaConf.save(cfg, cfg_path)


def get_hydra_dir() -> Path:
    """Get hydra dir."""
    hydra_cfg = HydraConfig.get()
    return Path(hydra_cfg.runtime.output_dir)


def get_hydra_timestamp() -> str:
    """Get hydra timestamp, specifying log folder version."""
    hydra_dir = get_hydra_dir()
    return hydra_dir.stem
