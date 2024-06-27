"""Command line interface to run SSL pretraining.

From the project root directory, do:

    poetry run python main.py group.field=value
"""

from hydra_zen import store, zen
from pytorch_lightning import seed_everything

import conf.data
import conf.experiment
import conf.hydra_conf
import conf.model
import conf.opt
import conf.run
import conf.trainer  # noqa: F401
from ssl_models.run_experiment import run_experiment

# %%
if __name__ == "__main__":
    store.add_to_hydra_store(overwrite_ok=True)
    zen_f = zen(
        run_experiment,
        pre_call=[
            lambda cfg: (
                seed_everything(cfg.run.seed, workers=True)
                if cfg.run.reproducible
                else lambda: None
            ),
        ],
    )
    zen_f.hydra_main(config_name="base_ssl_experiment", version_base="1.3")
