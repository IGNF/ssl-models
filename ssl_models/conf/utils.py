"""Utils to convert between configs and dicts."""

from dataclasses import asdict

from dacite.core import from_dict
from omegaconf import DictConfig, ListConfig, OmegaConf

from ssl_models.conf.dataset.utils import DATASETS
from ssl_models.conf.datasets import DatasetsConfig
from ssl_models.conf.mask import MaskConfig

DATASETS_GLOBAL_ARGS = ("root_dir", "name_datasets", "sample_datasets")


def datasets_from_dict(datasets_dict: dict) -> DatasetsConfig:
    """Convert dict to datasets config."""
    kwargs = {
        name_arg: datasets_dict[name_arg]
        for name_arg in DATASETS_GLOBAL_ARGS
        if name_arg in datasets_dict
    }
    for name_dataset, type_dataset in DATASETS.items():
        if name_dataset in datasets_dict:
            kwargs[name_dataset] = from_dict(
                data_class=type_dataset,
                data=datasets_dict[name_dataset],
            )

    return DatasetsConfig(**kwargs)


def datasets_to_dict(datasets: DatasetsConfig) -> dict:
    """Convert datasets config to dict."""
    datasets_dict = {
        name_arg: getattr(datasets, name_arg) for name_arg in DATASETS_GLOBAL_ARGS
    }
    for name_dataset in DATASETS:
        if name_dataset in datasets.__dict__:
            datasets_dict[name_dataset] = asdict(getattr(datasets, name_dataset))
    return datasets_dict


def mask_from_dict(mask_dict: dict) -> MaskConfig:
    """Convert dict to mask config."""
    return from_dict(data_class=MaskConfig, data=mask_dict)


def mask_to_dict(mask: MaskConfig) -> dict:
    """Convert mask config to dict."""
    return asdict(mask)


def json_default(obj: object) -> object:
    """Serialize objects not handled by json, converting omegaconf containers."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return str(obj)
