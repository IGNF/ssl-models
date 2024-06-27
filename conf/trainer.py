"""Trainer conf."""

from dataclasses import dataclass

from hydra_zen import store

from conf import pbuilds
from ssl_models.trainer import SSLTrainer


@dataclass
class TrainerConfig:  # noqa: D101
    accelerator: str = "auto"
    devices: str | int = "auto"
    strategy: str = "ddp_find_unused_parameters_true"
    precision: str = "bf16-mixed"


# Partial build, that will require instantiation in run_experiment
trainer_build = pbuilds(SSLTrainer, builds_bases=(TrainerConfig,))

trainer_store = store(group="trainer")
trainer_store(trainer_build, name="default_trainer")
