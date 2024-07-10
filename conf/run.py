"""Run conf."""  # noqa: EXE002

from dataclasses import dataclass, field

from hydra_zen import MISSING, store


@dataclass
class RunConfig:  # noqa: D101
    exp_dir: str = MISSING
    exp_name: str = MISSING
    probe: bool = True
    finetune: bool = True
    reproducible: bool = True
    seed: int = 42
    logged_images_per_epoch: int = 10
    use_clearml: bool = True
    clearml_project_name: str = "ssl"
    clearml_tags: list[str] = field(default_factory=lambda: ["geopile", "hydra"])
    clearml_offline_mode: bool = False


run_store = store(group="run")
run_store(RunConfig, name="default_run")
