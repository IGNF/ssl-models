"""Module for experiment runs."""

from clearml import Task
from hydra_zen.typing import Partial

from conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig
from conf.run import RunConfig
from ssl_models import LOGGER
from ssl_models.geopile_data import GeoPileData
from ssl_models.model import SSLModule
from ssl_models.trainer import SSLTrainer


def run_experiment(
    run: RunConfig,
    opt_pretrain: OptPretrainConfig,
    opt_probe: OptProbeConfig,
    opt_finetune: OptFinetuneConfig,
    data: Partial[GeoPileData],
    model: Partial[SSLModule],
    trainer: Partial[SSLTrainer],
) -> None:
    """Run experiment, with arguments set up using hydra zen."""
    if run.load_ckpt_path:
        model = SSLModule.load_from_checkpoint(run.load_ckpt_path)
    else:
        model = model()

    data_pretrain = data(ssl_phase="pretrain")
    trainer_pretrain = trainer(
        run=run,
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
            "data": vars(data_pretrain),
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
        data_probe = data(ssl_phase="probe")
        trainer_probe = trainer(
            run=run,
            ssl_phase="probe",
            opt=opt_probe,
        )
        trainer_probe.fit(model, datamodule=data_probe)

    if opt_finetune.epochs > 0:
        LOGGER.info("Finetune")
        data_finetune = data(ssl_phase="finetune")
        trainer_finetune = trainer(
            run=run,
            ssl_phase="finetune",
            opt=opt_finetune,
        )
        trainer_finetune.fit(model, datamodule=data_finetune)
