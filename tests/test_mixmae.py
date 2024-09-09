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
    "model_size, image_size, patch_size, num_levels, loss_fn, norm_pix_loss, nb_mix_images",
    itertools.product(
        ("tiny",),
        (128,),
        (4, 8),
        (3, 4),
        (torch.square, torch.abs),
        (True, False),
        (2, 4),
    ),
)
def test_mixmae(
    model_map,
    model_size,
    image_size,
    patch_size,
    num_levels,
    loss_fn,
    norm_pix_loss,
    nb_mix_images,
):
    model_args = {
        "in_chans": 3,
        "nb_mix_images": nb_mix_images,
        "image_size": image_size,
        "patch_size": patch_size,
        "out_patch_size": patch_size * 2 ** (num_levels - 1),
        "unpool": False,
        "num_levels": num_levels,
        "swin_version": 2,
        "v2_attn": True,
        "type_head": "linear",
        "loss_fn": loss_fn,
        "norm_pix_loss": norm_pix_loss,
        "fac_abs_enc": 1.0,
    }
    model_map[model_size](**model_args)
