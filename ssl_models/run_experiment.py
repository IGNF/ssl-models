"""Module for experiment runs."""

from functools import partial

from clearml import Task

from conf.data import DataConfig
from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig
from conf.model import BaselineConfig, ModelConfig
from conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig
from conf.run import RunConfig
from conf.trainer import TrainerConfig
from ssl_models import LOGGER
from ssl_models.train.baseline import BaselineModule
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
    mask: MaskConfig,
    model: ModelConfig | BaselineConfig,
    trainer: TrainerConfig,
) -> None:
    """Run experiment, with arguments set up using hydra zen."""
    trainer_partial = partial(SSLTrainer, run=run, **vars(trainer))
    data_partial = partial(SSLDataModule, datasets=datasets, **vars(data))
    load_partial = partial(
        SSLModule.load_from_checkpoint,
        map_location=lambda storage, _: storage,
        datasets=datasets,
    )

    if isinstance(model, BaselineConfig):
        model = BaselineModule(datasets=datasets, **vars(model))
    else:
        model = SSLModule(datasets=datasets, mask=mask, **vars(model))

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
            "datasets": vars(datasets),
            "data": vars(data),
            "mask": vars(mask),
            "model": vars(model),
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

    LOGGER.info("Set restore path")
    current_path = None
    load_path = run.load_ckpt_path if run.load_ckpt_path else None

    if opt_pretrain.epochs > 0:
        LOGGER.info("Pretrain")
        data_pretrain = data_partial(ssl_phase="pretrain", opt=opt_pretrain)
        trainer_pretrain = trainer_partial(
            ssl_phase="pretrain",
            opt=opt_pretrain,
        )
        current_path = trainer_pretrain.fit_and_test(
            model=model,
            datamodule=data_pretrain,
            load_partial=load_partial,
            current_path=current_path,
            load_path=load_path,
        )
        load_path = current_path

    if opt_probe.epochs > 0:
        LOGGER.info("Probe")
        data_probe = data_partial(ssl_phase="probe", opt=opt_probe)
        trainer_probe = trainer_partial(
            ssl_phase="probe",
            opt=opt_probe,
        )
        current_path = trainer_probe.fit_and_test(
            model=model,
            datamodule=data_probe,
            load_partial=load_partial,
            current_path=current_path,
            load_path=load_path,
        )

    if opt_finetune.epochs > 0:
        LOGGER.info("Finetune")
        data_finetune = data_partial(ssl_phase="finetune", opt=opt_finetune)
        trainer_finetune = trainer_partial(
            ssl_phase="finetune",
            opt=opt_finetune,
        )
        current_path = trainer_finetune.fit_and_test(
            model=model,
            datamodule=data_finetune,
            load_partial=load_partial,
            current_path=current_path,
            load_path=load_path,
        )
