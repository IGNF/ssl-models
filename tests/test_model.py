import itertools

import pytest

from ssl_models.model import SSLModule


@pytest.mark.parametrize(
    "stage, model_size",
    itertools.product(("train", "val", "test", "predict"), ("small",)),
)
def test_ssl_mae(stage, model_size):
    ssl_module = SSLModule(model="mae", model_size=model_size)
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "stage, model_size, swin_version, mask_batch",
    itertools.product(
        ("train", "val", "test", "predict"), ("small",), (1, 2), (True, False)
    ),
)
def test_ssl_mixmae(stage, model_size, swin_version, mask_batch):
    ssl_module = SSLModule(
        model="mixmae",
        model_size=model_size,
        swin_version=swin_version,
        mask_batch=mask_batch,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.xfail
def test_upsample_should_fail():
    _ = SSLModule(model="MAE", model_size="small", patch_size=14, upsample="unpool")
