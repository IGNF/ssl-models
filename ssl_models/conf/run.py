"""Run conf."""  # noqa: EXE002

from dataclasses import dataclass, field

from hydra_zen import MISSING, store


@dataclass
class RunConfig:  # noqa: D101
    exp_dir: str = MISSING
    exp_name: str = MISSING
    exp_uuid: str | None = None
    load_name: str | None = None
    load_phase: str = "pretrain"
    load_uuid: str | None = None
    load_ckpt_path: str | None = None
    fit_name: str | None = None
    fit_phase: str = "pretrain"
    fit_uuid: str | None = None
    fit_ckpt_path: str | None = None
    save_weights_only: bool = True
    export_weights: bool = False
    reproducible: bool = True
    seed: int = 42
    logged_images_per_epoch: int = 5
    log_per_class_metrics: bool = False
    per_class_metrics: list[str] = field(
        default_factory=lambda: ["iou", "f1", "precision", "recall"],
    )
    per_class_stages: list[str] = field(default_factory=lambda: ["val", "test"])
    log_confusion_matrix: bool = True
    use_clearml: bool = True
    clearml_project_name: str = "ssl"
    clearml_tags: list[str] = field(default_factory=lambda: ["multimodal", "hydra"])
    clearml_offline_mode: bool = False


run_store = store(group="run")
run_store(
    RunConfig,
    exp_uuid=None,
    load_uuid=None,
    load_ckpt_path=None,
    name="default_run",
)
