import json

import torch

from ssl_models.conf.utils import datasets_from_dict, mask_from_dict
from ssl_models.ssl.mae import MAE
from ssl_models.train.model import SSLModule
from tests import TEST_DATA_DIR


def test_load_mae():
    with (TEST_DATA_DIR / "model" / "config.json").open() as f:
        cfg = json.load(f)
    cfg.update(
        {
            "datasets": datasets_from_dict(cfg["datasets"]),
            "mask": mask_from_dict(cfg["mask"]),
        }
    )
    model = MAE(**cfg)

    state_dict = torch.load(TEST_DATA_DIR / "model" / "mae.pth", weights_only=True)
    model.load_state_dict(state_dict)


def test_export_load_mae(datasets_treesat, mask, tmp_path):
    ssl_module = SSLModule(
        datasets=datasets_treesat,
        mask=mask,
        fusion_mode="mod",
        inter_depth=0,
        model="mae",
        model_size="tiny",
    )
    ssl_module.export(tmp_path)

    with (tmp_path / "config.json").open() as f:
        cfg = json.load(f)
    cfg.update(
        {
            "datasets": datasets_from_dict(cfg["datasets"]),
            "mask": mask_from_dict(cfg["mask"]),
        }
    )
    model = MAE(**cfg)

    state_dict = torch.load(tmp_path / "mae.pth", weights_only=True)
    model.load_state_dict(state_dict)
    for name_param, param in model.state_dict().items():
        assert torch.equal(param, ssl_module.model.state_dict()[name_param])
