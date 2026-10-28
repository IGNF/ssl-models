"""Dataset class."""

import importlib
import pkgutil

from hydra_zen import MISSING, builds, store

import ssl_models.conf.dataset as dataset_package
from ssl_models.conf.dataset.utils import DATASETS, DatasetConfig


def _import_dataset_configs() -> None:
    """Import every dataset config module so that it registers itself."""
    for module in pkgutil.iter_modules(dataset_package.__path__):
        if module.name != "utils":
            importlib.import_module(f"{dataset_package.__name__}.{module.name}")


_import_dataset_configs()


# It does not seems possible to have nested dataclasses at all config levels,
# so we do not use a dataclass at the top level.
class DatasetsConfig:
    """Dataset config."""

    def __init__(
        self,
        root_dir: str,
        name_datasets: list[str],
        sample_datasets: list[float] | None = None,
        **datasets_conf: DatasetConfig,
    ) -> None:
        """Select registered datasets and assign their configs as attributes."""
        invalid = [
            name for name in (*name_datasets, *datasets_conf) if name not in DATASETS
        ]
        if invalid:
            msg = f"Invalid dataset names {invalid}. Registered: {sorted(DATASETS)}."
            raise ValueError(msg)

        if sample_datasets is None:
            sample_datasets = [1.0] * len(name_datasets)

        if len(name_datasets) != len(sample_datasets):
            msg = "Number of datasets and sampling weights are inconsistent."
            raise ValueError(msg)

        self.root_dir = root_dir
        self.name_datasets = name_datasets
        self.sample_datasets = sample_datasets

        self.datasets = {}
        self.sample = {}
        for name_dataset, sample_dataset in zip(name_datasets, sample_datasets):
            dataset = datasets_conf.get(name_dataset) or DATASETS[name_dataset]()
            setattr(self, name_dataset, dataset)
            self.datasets[name_dataset] = dataset
            self.sample[name_dataset] = sample_dataset


dataset_build = builds(
    DatasetsConfig,
    root_dir=MISSING,
    name_datasets=MISSING,
    sample_datasets=None,
    **{name: config_class() for name, config_class in DATASETS.items()},
)

dataset_store = store(group="datasets")
dataset_store(dataset_build, name="default_datasets")
