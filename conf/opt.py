"""Optimizer options conf."""

from dataclasses import dataclass

from hydra_zen import store


@dataclass
class OptPretrainConfig:  # noqa: D101
    base_lr: float = 3e-5
    b1: float = 0.9
    b2: float = 0.99
    wd: float = 0.01
    epochs: int = 20


pretrain_store = store(group="opt_pretrain")
pretrain_store(OptPretrainConfig, name="default_pretrain")


@dataclass
class OptProbeConfig:  # noqa: D101
    base_lr: float = 1e-5
    b1: float = 0.9
    b2: float = 0.99
    wd: float = 0.01
    epochs: int = 10


probe_store = store(group="opt_probe")
probe_store(OptProbeConfig, name="default_probe")


@dataclass
class OptFinetuneConfig:  # noqa: D101
    base_lr: float = 1e-5
    b1: float = 0.9
    b2: float = 0.99
    wd: float = 0.01
    epochs: int = 20


finetune_store = store(group="opt_finetune")
finetune_store(OptFinetuneConfig, name="default_finetune")
