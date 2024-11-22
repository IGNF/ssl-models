import itertools

import pytest
import torch

from ssl_models.ssl.mae import mae_large, mae_medium, mae_small, mae_tiny


@pytest.fixture()
def model_map():
    return {
        "tiny": mae_tiny,
        "small": mae_small,
        "medium": mae_medium,
        "large": mae_large,
    }


@pytest.mark.parametrize(
    "model_size, mask_ratio, unpool, type_head, norm_pix_loss, fac_abs_enc",
    itertools.product(
        ("tiny",),
        (0, 0.75),
        (True, False),
        ("linear", "attentive"),
        (True, False),
        (0.0, 1.0),
    ),
)
def test_mae(
    datasets_geopile,
    model_map,
    model_size,
    mask_ratio,
    unpool,
    type_head,
    norm_pix_loss,
    fac_abs_enc,
):
    model_args = {
        "datasets": datasets_geopile,
        "model": "mae",
        "nb_mix_images": None,
        "mask_ratio": mask_ratio,
        "unpool": unpool,
        "num_levels": 1,
        "type_head": type_head,
        "loss_fn": torch.square,
        "norm_pix_loss": norm_pix_loss,
        "fac_abs_enc": fac_abs_enc,
    }
    model_map[model_size](**model_args)
