"""Utils to convert between configs and dicts."""

from dataclasses import asdict, is_dataclass

from dacite.core import from_dict

from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig


def datasets_from_dict(datasets_dict: dict) -> DatasetsConfig:
    """Convert dict to datasets config."""
    datasets = DatasetsConfig(
        **{
            name_arg: val_arg
            for name_arg, val_arg in datasets_dict.items()
            if not isinstance(val_arg, dict)
        },
    )
    for name_arg, val_arg in datasets.kwargs.items():
        if not is_dataclass(val_arg):
            continue
        setattr(
            datasets,
            name_arg,
            from_dict(data_class=type(val_arg), data=datasets_dict[name_arg]),
        )
    return datasets


def datasets_to_dict(datasets: DatasetsConfig) -> dict:
    """Convert datasets config to dict."""
    datasets_dict = {
        name_arg: val_arg
        for name_arg, val_arg in datasets.kwargs.items()
        if not is_dataclass(val_arg)
    }
    for name_arg, val_arg in datasets.kwargs.items():
        if not is_dataclass(val_arg):
            continue
        datasets_dict[name_arg] = asdict(val_arg)
    return datasets_dict


def mask_from_dict(mask_dict: dict) -> MaskConfig:
    """Convert dict to mask config."""
    return from_dict(data_class=MaskConfig, data=mask_dict)


def mask_to_dict(mask: MaskConfig) -> dict:
    """Convert mask config to dict."""
    return asdict(mask)
