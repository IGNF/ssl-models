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
    "model_size, mask_ratio, image_size, patch_size, norm_pix_loss, fac_abs_enc",
    itertools.product(
        ("tiny",), (0, 0.75), (128, 256), (16, 32), (True, False), (0.0, 1.0)
    ),
)
def test_mae(
    model_map,
    model_size,
    mask_ratio,
    image_size,
    patch_size,
    norm_pix_loss,
    fac_abs_enc,
):
    model_args = {
        "in_chans": 3,
        "nb_mix_images": None,
        "mask_ratio": mask_ratio,
        "image_size": image_size,
        "patch_size": patch_size,
        "out_patch_size": patch_size,
        "unpool": True,
        "num_levels": 1,
        "swin_version": None,
        "v2_attn": None,
        "type_head": "linear",
        "loss_fn": torch.square,
        "norm_pix_loss": norm_pix_loss,
        "fac_abs_enc": fac_abs_enc,
    }
    model_map[model_size](**model_args)
