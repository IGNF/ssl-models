import itertools

import pytest
import torch

from ssl_models.ssl.mixmae import (
    mixmae_base,
    mixmae_huge,
    mixmae_large,
    mixmae_small,
    mixmae_tiny,
)


@pytest.fixture
def model_map():
    return {
        "tiny": mixmae_tiny,
        "small": mixmae_small,
        "base": mixmae_base,
        "large": mixmae_large,
        "huge": mixmae_huge,
    }


@pytest.mark.parametrize(
    "model_size, swin_version, num_levels, loss_fn, norm_pix_loss, nb_mix_images",
    itertools.product(
        ("tiny",),
        (1, 2),
        (3, 4),
        (torch.square, torch.abs),
        (True, False),
        (2, 4),
    ),
)
def test_mixmae(
    datasets_geopile,
    model_map,
    model_size,
    swin_version,
    num_levels,
    loss_fn,
    norm_pix_loss,
    nb_mix_images,
):
    model_args = {
        "datasets": datasets_geopile,
        "model": "mixmae",
        "nb_mix_images": nb_mix_images,
        "unpool": False,
        "num_levels": num_levels,
        "swin_version": swin_version,
        "v2_attn": True,
        "type_head": "linear",
        "loss_fn": loss_fn,
        "norm_pix_loss": norm_pix_loss,
        "fac_abs_enc": 1.0,
    }
    model_map[model_size](**model_args)
