import itertools

import pytest

from ssl_models.mixmae import (
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
    "model_size, image_size, patch_size, mask_batch, swin_version, norm_pix_loss, nb_mix_images",
    itertools.product(
        ("tiny",),
        (128, 256),
        (4, 8),
        (True, False),
        (1, 2),
        (True, False),
        (2, 4),
    ),
)
def test_mixmae(
    model_map,
    model_size,
    image_size,
    patch_size,
    mask_batch,
    swin_version,
    norm_pix_loss,
    nb_mix_images,
):
    model_args = {
        "image_size": image_size,
        "patch_size": patch_size,
        "mask_batch": mask_batch,
        "swin_version": swin_version,
        "norm_pix_loss": norm_pix_loss,
        "nb_mix_images": nb_mix_images,
    }
    model_map[model_size](**model_args)
