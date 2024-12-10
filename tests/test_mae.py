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
    "model_size, multimodal",
    itertools.product(
        ("tiny",),
        ("msgfm", "shared", "mod", "group"),
    ),
)
def test_mae(datasets_geopile, mask, model_map, model_size, multimodal):
    model_args = {
        "datasets": datasets_geopile,
        "mask": mask,
        "multimodal": multimodal,
        "model": "mae",
        "unpool": False,
        "num_levels": 1,
        "type_head": "attentive",
        "loss_fn": torch.square,
        "norm_pix_loss": True,
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map[model_size](**model_args)


@pytest.mark.parametrize(
    "model_size, loss_fn, norm_pix_loss",
    itertools.product(
        ("tiny",),
        (torch.square, torch.abs),
        (True, False),
    ),
)
def test_mae_loss(
    datasets_geopile,
    mask,
    model_map,
    model_size,
    loss_fn,
    norm_pix_loss,
):
    model_args = {
        "datasets": datasets_geopile,
        "mask": mask,
        "multimodal": "mod",
        "model": "mae",
        "unpool": False,
        "num_levels": 1,
        "type_head": "attentive",
        "loss_fn": loss_fn,
        "norm_pix_loss": norm_pix_loss,
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map[model_size](**model_args)


@pytest.mark.parametrize(
    "model_size, unpool, type_head, fac_abs_enc, fac_date_enc",
    itertools.product(
        ("tiny",),
        (True, False),
        ("attentive", "linear"),
        (1.0, 0.0),
        (1.0, 0.0),
    ),
)
def test_mae_arch(
    datasets_geopile,
    mask,
    model_map,
    model_size,
    unpool,
    type_head,
    fac_abs_enc,
    fac_date_enc,
):
    model_args = {
        "datasets": datasets_geopile,
        "mask": mask,
        "multimodal": "mod",
        "model": "mae",
        "unpool": unpool,
        "num_levels": 1,
        "type_head": type_head,
        "loss_fn": torch.square,
        "norm_pix_loss": True,
        "fac_abs_enc": fac_abs_enc,
        "fac_date_enc": fac_date_enc,
    }
    model_map[model_size](**model_args)
