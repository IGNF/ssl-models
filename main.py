"""Command line interface to run SSL pretraining.

From the project root directory, do:

    poetry run python main.py group.field=value
"""

from hydra_zen import store, zen

import ssl_models.conf.data
import ssl_models.conf.dataset
import ssl_models.conf.experiment
import ssl_models.conf.hydra_conf
import ssl_models.conf.model
import ssl_models.conf.opt
import ssl_models.conf.run
import ssl_models.conf.trainer  # noqa: F401
from ssl_models.hydra_utils import pre_call_resolve, pre_call_seed
from ssl_models.run_experiment import run_experiment

# %%
if __name__ == "__main__":
    store.add_to_hydra_store(overwrite_ok=True)
    zen_f = zen(run_experiment, pre_call=[pre_call_seed, pre_call_resolve])
    zen_f.hydra_main(config_name="base_ssl_experiment", version_base="1.3")
