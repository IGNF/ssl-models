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
    "model_size, image_size, mask_batch, swin_version",
    itertools.product(
        ("tiny",),
        (128, 256),
        (True, False),
        (1, 2),
    ),
)
def test_mixmae(model_map, model_size, image_size, mask_batch, swin_version):
    model_args = {
        "image_size": image_size,
        "mask_batch": mask_batch,
        "swin_version": swin_version,
    }
    model_map[model_size](**model_args)
