import pytest
import torch
from pytorch_lightning import Trainer

from ssl_models.conf.augment import DinoAugmentConfig
from ssl_models.conf.loss import DINOv3LossConfig
from ssl_models.conf.model import DINOv3Config, MiMConfig
from ssl_models.train.lit_dino import LitDINOv3
from ssl_models.train.load import build_module
from ssl_models.train.lit_mim import LitMiM


@pytest.mark.parametrize(
    "model, module_class",
    [
        (MiMConfig(model_size="tiny"), LitMiM),
        (
            DINOv3Config(
                model_size="tiny",
                loss=DINOv3LossConfig(
                    dino_n_prototypes=128,
                    ibot_n_prototypes=128,
                    head_hidden_dim=64,
                    head_bottleneck_dim=32,
                ),
            ),
            LitDINOv3,
        ),
    ],
)
def test_build_and_reload_module(model, module_class, dataset_treesat, tmp_path):
    augment = DinoAugmentConfig(n_local_crops=2)
    module = build_module(model, dataset_treesat, augment)
    assert type(module) is module_class

    ckpt_path = tmp_path / "module.ckpt"
    trainer = Trainer(logger=False, enable_checkpointing=False, accelerator="cpu")
    trainer.strategy.connect(module)
    trainer.save_checkpoint(ckpt_path)

    reloaded = build_module(model, dataset_treesat, augment, ckpt_path=ckpt_path)
    assert type(reloaded) is module_class
    state_dict, reloaded_state_dict = module.state_dict(), reloaded.state_dict()
    assert state_dict.keys() == reloaded_state_dict.keys()
    for key, value in state_dict.items():
        assert torch.equal(value, reloaded_state_dict[key]), key
