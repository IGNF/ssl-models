"""Init conf."""

import importlib
import pkgutil

from hydra_zen import make_custom_builds_fn

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=False)


def register_configs(package: str = __name__) -> None:
    """Recursively import every module of `package`, registering its configs.

    Configs register themselves in the hydra store when their module is imported.
    This is not done at package import, which would create circular imports
    (e.g. `ssl_models.dataset` modules import `conf.dataset.utils`).
    """
    path = importlib.import_module(package).__path__
    for module in pkgutil.walk_packages(path, prefix=f"{package}."):
        importlib.import_module(module.name)
