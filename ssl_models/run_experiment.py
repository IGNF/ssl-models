"""Module for experiment runs."""

from dataclasses import asdict
from functools import partial

from clearml import Task

from ssl_models import LOGGER
from ssl_models.conf.augment import DinoAugmentConfig
from ssl_models.conf.data import DataConfig
from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.model import BaselineConfig, DINOv3Config, MiMConfig
from ssl_models.conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig
from ssl_models.conf.run import RunConfig
from ssl_models.conf.trainer import TrainerConfig
from ssl_models.train.data import SSLDataModule
from ssl_models.train.load import build_module
from ssl_models.train.trainer import SSLTrainer


def run_experiment(
    run: RunConfig,
    opt_pretrain: OptPretrainConfig,
    opt_probe: OptProbeConfig,
    opt_finetune: OptFinetuneConfig,
    dataset: DatasetConfig,
    data: DataConfig,
    augment: DinoAugmentConfig,
    model: MiMConfig | DINOv3Config | BaselineConfig,
    trainer: TrainerConfig,
) -> None:
    """Run experiment, with arguments set up using hydra zen."""
    trainer_partial = partial(SSLTrainer, run=run, **vars(trainer))
    data_partial = partial(SSLDataModule, dataset=dataset, **vars(data))

    if isinstance(model, DINOv3Config) and (
        opt_probe.epochs > 0 or opt_finetune.epochs > 0
    ):
        msg = (
            "DINO only supports pretraining for now: "
            "set opt_probe.epochs=0 and opt_finetune.epochs=0."
        )
        raise NotImplementedError(msg)

    if (
        run.use_clearml
        and trainer_partial(ssl_phase="pretrain", opt=opt_pretrain).global_rank == 0
    ):
        LOGGER.info("Set up ClearML")
        config = {
            "run": vars(run),
            "opt_pretrain": vars(opt_pretrain),
            "opt_probe": vars(opt_probe),
            "opt_finetune": vars(opt_finetune),
            "dataset": asdict(dataset),
            "data": vars(data),
            "augment": vars(augment),
            "model": asdict(model),
            "trainer": vars(trainer),
        }
        if run.clearml_offline_mode:
            Task.set_offline(offline_mode=True)
        task = Task.init(
            project_name=run.clearml_project_name,
            task_name=run.exp_name,
            tags=run.clearml_tags,
        )
        task.set_parameters_as_dict(config)

    module = build_module(model, dataset, augment, ckpt_path=run.load_ckpt_path)

    if (
        opt_pretrain.epochs > 0
        and not run.load_ckpt_path
        and (not run.fit_ckpt_path or run.fit_phase == "pretrain")
    ):
        LOGGER.info("Pretrain")
        data_pretrain = data_partial(ssl_phase="pretrain", opt=opt_pretrain)
        trainer_pretrain = trainer_partial(
            ssl_phase="pretrain",
            opt=opt_pretrain,
        )
        trainer_pretrain.fit_and_test(
            model=module,
            datamodule=data_pretrain,
            ckpt_path=run.fit_ckpt_path,
        )

    if (
        opt_probe.epochs > 0
        and (not run.load_ckpt_path or run.load_phase == "pretrain")
        and (not run.fit_ckpt_path or run.fit_phase == "probe")
    ):
        LOGGER.info("Probe")
        data_probe = data_partial(ssl_phase="probe", opt=opt_probe)
        trainer_probe = trainer_partial(
            ssl_phase="probe",
            opt=opt_probe,
        )
        trainer_probe.fit_and_test(
            model=module,
            datamodule=data_probe,
            ckpt_path=run.fit_ckpt_path,
        )

    if opt_finetune.epochs > 0:
        LOGGER.info("Finetune")
        data_finetune = data_partial(ssl_phase="finetune", opt=opt_finetune)
        trainer_finetune = trainer_partial(
            ssl_phase="finetune",
            opt=opt_finetune,
        )
        trainer_finetune.fit_and_test(
            model=module,
            datamodule=data_finetune,
            ckpt_path=run.fit_ckpt_path,
        )
