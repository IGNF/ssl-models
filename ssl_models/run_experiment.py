from clearml import Task
from hydra_zen.typing import Partial

from ssl_models import LOGGER
from ssl_models.geopile_data import GeoPileData
from ssl_models.model import SSLModule
from ssl_models.trainer import SSLTrainer


def run_experiment(
    run,
    opt_pretrain,
    opt_probe,
    opt_finetune,
    data: GeoPileData,
    model: SSLModule,
    trainer: Partial[SSLTrainer],
):
    trainer_pretrain = trainer(
        run=run,
        ssl_phase="pretrain",
        opt=opt_pretrain,
    )
    trainer_probe = trainer(
        run=run,
        ssl_phase="probe",
        opt=opt_probe,
    )
    trainer_finetune = trainer(
        run=run,
        ssl_phase="finetune",
        opt=opt_finetune,
    )

    if run.use_clearml and trainer_pretrain.global_rank == 0:
        LOGGER.info("Set up ClearML")
        config = dict(
            run=vars(run),
            opt_pretrain=vars(opt_pretrain),
            opt_probe=vars(opt_probe),
            opt_finetune=vars(opt_finetune),
            data=vars(data),
            model=vars(model),
            trainer=vars(trainer_pretrain),
        )
        if run.clearml_offline_mode:
            Task.set_offline(offline_mode=True)
        task = Task.init(
            project_name=run.clearml_project_name,
            task_name=run.exp_name,
            tags=run.clearml_tags,
        )
        task.set_parameters_as_dict(config)

    LOGGER.info("Pretrain")
    trainer_pretrain.fit(model, datamodule=data)

    if run.probe > 0:
        LOGGER.info("Probe")
        trainer_probe.fit(model, datamodule=data)

    if run.finetune > 0:
        LOGGER.info("Finetune")
        trainer_finetune.fit(model, datamodule=data)
