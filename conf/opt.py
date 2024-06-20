from dataclasses import dataclass

from hydra_zen import store

# ================== OptPretrain Conf==================
pretrain_store = store(group="opt_pretrain")


@dataclass
class OptPretrainConfig:
    base_lr: float = 3e-5
    b1: float = 0.9
    b2: float = 0.99
    wd: float = 0.01
    max_epochs: int = 20


pretrain_store(OptPretrainConfig, name="default_pretrain")


# ================== OptProbe Conf==================
probe_store = store(group="opt_probe")


@dataclass
class OptProbeConfig:
    base_lr: float = 1e-5
    b1: float = 0.9
    b2: float = 0.99
    wd: float = 0.01
    max_epochs: int = 10


probe_store(OptProbeConfig, name="default_probe")


# ================== OptFinetune Conf==================
finetune_store = store(group="opt_finetune")


@dataclass
class OptFinetuneConfig:
    base_lr: float = 1e-5
    b1: float = 0.9
    b2: float = 0.99
    wd: float = 0.01
    max_epochs: int = 10


finetune_store(OptFinetuneConfig, name="default_finetune")
