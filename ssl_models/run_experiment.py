"""Module for experiment runs."""

from functools import partial

from clearml import Task

from ssl_models import LOGGER
from ssl_models.conf.augment import DinoAugmentConfig
from ssl_models.conf.data import DataConfig
from ssl_models.conf.datasets import DatasetsConfig
from ssl_models.conf.loss import DinoLossConfig, MiMLossConfig
from ssl_models.conf.mask import DinoMaskConfig, MiMMaskConfig
from ssl_models.conf.model import BaselineConfig, DinoConfig, MiMConfig
from ssl_models.conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig
from ssl_models.conf.run import RunConfig
from ssl_models.conf.trainer import TrainerConfig
from ssl_models.train.baseline import BaselineModule
from ssl_models.train.data import SSLDataModule
from ssl_models.train.dino import DINOModule
from ssl_models.train.mim import MiMModule
from ssl_models.train.trainer import SSLTrainer


def check_configs(
    name: str,
    mask: object,
    mask_class: type,
    loss: object,
    loss_class: type,
) -> None:
    """Check that the mask and loss configs match the model."""
    for group, cfg, cfg_class in (
        ("mask", mask, mask_class),
        ("loss", loss, loss_class),
    ):
        if not isinstance(cfg, cfg_class):
            msg = (
                f"Invalid {group} config {type(cfg).__name__} for the {name} model: "
                f"set {group}={name}_{group}."
            )
            raise TypeError(msg)


def run_experiment(
    run: RunConfig,
    opt_pretrain: OptPretrainConfig,
    opt_probe: OptProbeConfig,
    opt_finetune: OptFinetuneConfig,
    datasets: DatasetsConfig,
    data: DataConfig,
    mask: MiMMaskConfig | DinoMaskConfig,
    loss: MiMLossConfig | DinoLossConfig,
    augment: DinoAugmentConfig,
    model: MiMConfig | BaselineConfig | DinoConfig,
    trainer: TrainerConfig,
) -> None:
    """Run experiment, with arguments set up using hydra zen."""
    trainer_partial = partial(SSLTrainer, run=run, **vars(trainer))
    data_partial = partial(SSLDataModule, datasets=datasets, **vars(data))

    if isinstance(model, DinoConfig):
        check_configs("dino", mask, DinoMaskConfig, loss, DinoLossConfig)
        if opt_probe.epochs > 0 or opt_finetune.epochs > 0:
            msg = (
                "DINO only supports pretraining for now: "
                "set opt_probe.epochs=0 and opt_finetune.epochs=0."
            )
            raise NotImplementedError(msg)
        dino_kwargs = {
            "datasets": datasets,
            "mask": mask,
            "loss": loss,
            "augment": augment,
        }
        model_partial = partial(DINOModule, **dino_kwargs, **vars(model))
        load_kwargs = dino_kwargs
        module_class = DINOModule
    elif isinstance(model, BaselineConfig):
        model_partial = partial(BaselineModule, datasets=datasets, **vars(model))
        load_kwargs = {"datasets": datasets, "mask": mask}
        module_class = BaselineModule
    else:
        check_configs("mim", mask, MiMMaskConfig, loss, MiMLossConfig)
        model_partial = partial(
            MiMModule, datasets=datasets, mask=mask, loss=loss, **vars(model)
        )
        load_kwargs = {"datasets": datasets, "mask": mask, "loss": loss}
        module_class = MiMModule

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
            "loss": vars(loss),
            "augment": vars(augment),
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

    if run.load_ckpt_path:
        model = module_class.load_from_checkpoint(
            run.load_ckpt_path,
            map_location=lambda storage, _: storage,
            strict=False,
            **load_kwargs,
        )
    else:
        model = model_partial()

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
            model=model,
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
            model=model,
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
            model=model,
            datamodule=data_finetune,
            ckpt_path=run.fit_ckpt_path,
        )
