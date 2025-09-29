import itertools

import pytest

from ssl_models.train.model import SSLModule


@pytest.mark.parametrize(
    "interpolate, multimodal, allmods_depth, stage, model_size",
    itertools.product(
        ("nearest",),
        ("msgfm", "shared", "mod", "group"),
        (0,),
        ("train", "val", "test"),
        ("tiny",),
    ),
)
def test_ssl_mae(
    datasets_geopile,
    mask,
    interpolate,
    multimodal,
    allmods_depth,
    stage,
    model_size,
):
    ssl_module = SSLModule(
        datasets=datasets_geopile,
        mask=mask,
        interpolate=interpolate,
        multimodal=multimodal,
        allmods_depth=allmods_depth,
        model="mae",
        model_size=model_size,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "interpolate, multimodal, allmods_depth, stage, model_size",
    itertools.product(
        ("nearest",),
        ("mod", "group"),
        (3,),
        ("train", "val", "test"),
        ("tiny",),
    ),
)
def test_ssl_allmods_mae(
    datasets_geopile,
    mask,
    interpolate,
    multimodal,
    allmods_depth,
    stage,
    model_size,
):
    ssl_module = SSLModule(
        datasets=datasets_geopile,
        mask=mask,
        interpolate=interpolate,
        multimodal=multimodal,
        allmods_depth=allmods_depth,
        model="mae",
        model_size=model_size,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "interpolate, multimodal, allmods_depth, stage, model_size, type_head, loss",
    itertools.product(
        ("nearest",),
        ("mod",),
        (0,),
        ("train", "val", "test"),
        ("tiny",),
        ("attentive", "linear"),
        ("l2_norm", "l2"),
    ),
)
def test_ssl_head_loss(
    datasets_geopile,
    mask,
    interpolate,
    multimodal,
    allmods_depth,
    stage,
    model_size,
    type_head,
    loss,
):
    ssl_module = SSLModule(
        datasets=datasets_geopile,
        mask=mask,
        interpolate=interpolate,
        multimodal=multimodal,
        allmods_depth=allmods_depth,
        model="mae",
        model_size=model_size,
        type_head=type_head,
        loss=loss,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "interpolate, multimodal, allmods_depth, stage, model_size, num_levels, swin_version, swin_abs_enc",
    itertools.product(
        ("nearest",),
        ("msgfm", "shared"),
        (0,),
        ("train",),
        ("tiny",),
        (3, 4),
        (1, 2),
        (True, False),
    ),
)
def test_ssl_mixmae(
    datasets_geopile,
    mask,
    interpolate,
    multimodal,
    allmods_depth,
    stage,
    model_size,
    num_levels,
    swin_version,
    swin_abs_enc,
):
    ssl_module = SSLModule(
        datasets=datasets_geopile,
        mask=mask,
        interpolate=interpolate,
        multimodal=multimodal,
        allmods_depth=allmods_depth,
        model="mixmae",
        model_size=model_size,
        swin_version=swin_version,
        num_levels=num_levels,
        swin_abs_enc=swin_abs_enc,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.xfail
def test_model_mod_mixmae_should_fail(datasets_geopile, mask):
    _ = SSLModule(
        datasets=datasets_geopile,
        mask=mask,
        multimodal="mod",
        allmods_depth=0,
        model="mixmae",
        model_size="tiny",
    )


@pytest.mark.xfail
def test_model_group_mixmae_should_fail(datasets_geopile, mask):
    _ = SSLModule(
        datasets=datasets_geopile,
        mask=mask,
        multimodal="group",
        allmods_depth=0,
        model="mixmae",
        model_size="tiny",
    )


@pytest.mark.xfail
def test_levels_should_fail(datasets_geopile, mask):
    _ = SSLModule(
        datasets=datasets_geopile,
        mask=mask,
        multimodal="shared",
        allmods_depth=0,
        model="mixmae",
        model_size="tiny",
        num_levels=1,
    )


@pytest.mark.xfail
def test_model_should_fail(datasets_geopile, mask):
    _ = SSLModule(
        datasets=datasets_geopile,
        mask=mask,
        multimodal="shared",
        allmods_depth=0,
        model="MAE",
        model_size="tiny",
    )


@pytest.mark.xfail
def test_missing_args_should_fail():
    _ = SSLModule(model="mae", model_size="tiny")
