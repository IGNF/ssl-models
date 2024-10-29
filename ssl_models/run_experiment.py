"""Module for experiment runs."""

from functools import partial

from clearml import Task

from conf.data import DataConfig
from conf.datasets import DatasetsConfig
from conf.model import ModelConfig
from conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig
from conf.run import RunConfig
from conf.trainer import TrainerConfig
from ssl_models import LOGGER
from ssl_models.train.data import SSLDataModule
from ssl_models.train.model import SSLModule
from ssl_models.train.trainer import SSLTrainer


def run_experiment(
    run: RunConfig,
    opt_pretrain: OptPretrainConfig,
    opt_probe: OptProbeConfig,
    opt_finetune: OptFinetuneConfig,
    datasets: DatasetsConfig,
    data: DataConfig,
    model: ModelConfig,
    trainer: TrainerConfig,
) -> None:
    """Run experiment, with arguments set up using hydra zen."""
    trainer_partial = partial(SSLTrainer, datasets=datasets, run=run, **vars(trainer))
    model_partial = partial(SSLModule, datasets=datasets, **vars(model))
    data_partial = partial(SSLDataModule, datasets=datasets, **vars(data))

    if run.load_ckpt_path:
        model = SSLModule.load_from_checkpoint(run.load_ckpt_path, datasets=datasets)
    else:
        model = model_partial()

    data_pretrain = data_partial(ssl_phase="pretrain")
    trainer_pretrain = trainer_partial(
        ssl_phase="pretrain",
        opt=opt_pretrain,
    )

    if run.use_clearml and trainer_pretrain.global_rank == 0:
        LOGGER.info("Set up ClearML")
        config = {
            "run": vars(run),
            "opt_pretrain": vars(opt_pretrain),
            "opt_probe": vars(opt_probe),
            "opt_finetune": vars(opt_finetune),
            "data": vars(data),
            "datasets": vars(datasets),
            "model": vars(model),
            "trainer": vars(trainer_pretrain),
        }
        if run.clearml_offline_mode:
            Task.set_offline(offline_mode=True)
        task = Task.init(
            project_name=run.clearml_project_name,
            task_name=run.exp_name,
            tags=run.clearml_tags,
        )
        task.set_parameters_as_dict(config)

    LOGGER.info("Pretrain")
    trainer_pretrain.fit(model, datamodule=data_pretrain)

    if opt_probe.epochs > 0:
        LOGGER.info("Probe")
        data_probe = data_partial(ssl_phase="probe")
        trainer_probe = trainer_partial(
            ssl_phase="probe",
            opt=opt_probe,
        )
        trainer_probe.fit(model, datamodule=data_probe)

    if opt_finetune.epochs > 0:
        LOGGER.info("Finetune")
        data_finetune = data_partial(ssl_phase="finetune")
        trainer_finetune = trainer_partial(
            ssl_phase="finetune",
            opt=opt_finetune,
        )
        trainer_finetune.fit(model, datamodule=data_finetune)
