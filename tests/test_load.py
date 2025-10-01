import json

import torch

from conf.utils import datasets_from_dict, mask_from_dict
from ssl_models.ssl.mae import MAE
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
