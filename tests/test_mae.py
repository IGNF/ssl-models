import itertools

import pytest

from ssl_models.mae import mae_large, mae_medium, mae_small, mae_tiny
from ssl_models.ssl_module import downsampling_from_patch_size


@pytest.fixture()
def model_map():
    return {
        "tiny": mae_tiny,
        "small": mae_small,
        "medium": mae_medium,
        "large": mae_large,
    }


@pytest.mark.parametrize(
    "model_size, mask_ratio, image_size, patch_size, unpool",
    itertools.product(("tiny",), (0, 0.75), (128, 256), (16, 32), (False, True)),
)
def test_mae(model_map, model_size, mask_ratio, image_size, patch_size, unpool):
    num_downsampling = downsampling_from_patch_size(unpool, patch_size)
    model_args = {
        "mask_ratio": mask_ratio,
        "image_size": image_size,
        "patch_size": patch_size,
        "num_downsampling": num_downsampling,
        "unpool": unpool,
    }
    model_map[model_size](**model_args)
