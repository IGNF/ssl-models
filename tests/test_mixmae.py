import itertools

import pytest

from ssl_models.mixmae import (
    mixmim_base,
    mixmim_huge,
    mixmim_large,
    mixmim_small,
    mixmim_tiny,
)


@pytest.fixture
def model_map():
    return {
        "tiny": mixmim_tiny,
        "small": mixmim_small,
        "base": mixmim_base,
        "large": mixmim_large,
        "huge": mixmim_huge,
    }


@pytest.mark.parametrize(
    "model_size, image_size, mask_batch",
    itertools.product(
        ("tiny",),
        (128, 256),
        (True, False)
    ),
)
def test_mixmae(model_map, model_size, image_size, mask_batch):
    model_args = {
        "image_size": image_size,
        "mask_batch": mask_batch,
    }
    model_map[model_size](**model_args)
