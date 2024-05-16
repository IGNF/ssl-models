"""
Command line interface to run SSL pretraining.

From the project root directory, do:

    poetry run python main.py fit --config=./config/conf.yaml

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""

from pytorch_lightning.cli import LightningCLI, SaveConfigCallback

from ssl_models.data_module import SSLDataModule
from ssl_models.ssl_module import SSLModule
from ssl_models.trainer import SSLTrainer

# %%
if __name__ == "__main__":
    cli = LightningCLI(
        datamodule_class=SSLDataModule,
        model_class=SSLModule,
        trainer_class=SSLTrainer,
        save_config_callback=SaveConfigCallback,
    )
