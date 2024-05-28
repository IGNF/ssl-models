import itertools

import pytest

from ssl_models.ssl_module import SSLModule


@pytest.mark.parametrize(
    "stage, patch_size, unpool",
    itertools.product(("train", "val", "test", "predict"), (16,), (False, True)),
)
def test_ssl_module(stage, patch_size, unpool):
    ssl_module = SSLModule(patch_size=patch_size, unpool=unpool)
    ssl_module.setup(stage=stage)


@pytest.mark.xfail
def test_should_fail():
    _ = SSLModule(patch_size=14, unpool=True)
