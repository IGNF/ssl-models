import itertools

import pytest

from ssl_models.ssl.model import SSLModule


@pytest.mark.parametrize(
    "stage, model_size",
    itertools.product(("train", "val", "test", "predict"), ("small",)),
)
def test_ssl_mae(stage, model_size):
    ssl_module = SSLModule(model="mae", model_size=model_size)
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "stage, model_size, swin_version, v2_attn, num_levels, swin_abs_enc",
    itertools.product(
        ("train", "val"),
        ("tiny",),
        (1, 2),
        (True, False),
        (3, 4),
        (True, False),
    ),
)
def test_ssl_mixmae(stage, model_size, swin_version, v2_attn, num_levels, swin_abs_enc):
    ssl_module = SSLModule(
        model="mixmae",
        model_size=model_size,
        swin_version=swin_version,
        v2_attn=v2_attn,
        num_levels=num_levels,
        swin_abs_enc=swin_abs_enc,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.xfail
def test_unpool_should_fail():
    _ = SSLModule(model="mae", model_size="tiny", patch_size=14, unpool=True)


@pytest.mark.xfail
def test_levels_should_fail():
    _ = SSLModule(model="mixmae", model_size="tiny", num_levels=1)


@pytest.mark.xfail
def test_model_should_fail():
    _ = SSLModule(model="MAE", model_size="tiny")
