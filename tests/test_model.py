import itertools

import pytest

from ssl_models.conf.loss import MiMLossConfig
from ssl_models.train.mim import MiMModule


@pytest.mark.parametrize(
    "fusion_mode, inter_depth, model_size, interpolate, stage",
    itertools.product(
        ("msgfm", "shared", "mod", "group"),
        (0,),
        ("tiny",),
        ("nearest",),
        ("train", "val", "test"),
    ),
)
def test_ssl_mae(
    dataset_treesat,
    mask,
    fusion_mode,
    inter_depth,
    model_size,
    interpolate,
    stage,
):
    ssl_module = MiMModule(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        model="mae",
        model_size=model_size,
        interpolate=interpolate,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "fusion_mode, inter_depth, model_size, interpolate, stage",
    itertools.product(
        ("mod", "group"),
        (3,),
        ("tiny",),
        ("nearest",),
        ("train", "val", "test"),
    ),
)
def test_ssl_inter_mae(
    dataset_treesat,
    mask,
    fusion_mode,
    inter_depth,
    model_size,
    interpolate,
    stage,
):
    ssl_module = MiMModule(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        model="mae",
        model_size=model_size,
        interpolate=interpolate,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "fusion_mode, inter_depth, model_size, type_head, interpolate, loss, stage",
    itertools.product(
        ("mod",),
        (0,),
        ("tiny",),
        ("attentive", "linear"),
        ("nearest",),
        ("l2_norm", "l2"),
        ("train", "val", "test"),
    ),
)
def test_ssl_head_loss(
    dataset_treesat,
    mask,
    fusion_mode,
    inter_depth,
    model_size,
    type_head,
    interpolate,
    loss,
    stage,
):
    ssl_module = MiMModule(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        model="mae",
        model_size=model_size,
        type_head=type_head,
        interpolate=interpolate,
        loss=MiMLossConfig(name=loss),
    )
    ssl_module.setup(stage=stage)


@pytest.mark.parametrize(
    "fusion_mode, inter_depth, model_size, interpolate, num_levels, swin_version, swin_abs_enc, stage,",
    itertools.product(
        ("msgfm", "shared"),
        (0,),
        ("tiny",),
        ("nearest",),
        (3, 4),
        (1, 2),
        (True, False),
        ("train",),
    ),
)
def test_ssl_mixmae(
    dataset_treesat,
    mask,
    fusion_mode,
    inter_depth,
    model_size,
    interpolate,
    num_levels,
    swin_version,
    swin_abs_enc,
    stage,
):
    ssl_module = MiMModule(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        model="mixmae",
        model_size=model_size,
        interpolate=interpolate,
        swin_version=swin_version,
        num_levels=num_levels,
        swin_abs_enc=swin_abs_enc,
    )
    ssl_module.setup(stage=stage)


@pytest.mark.xfail
def test_model_mod_mixmae_should_fail(dataset_treesat, mask):
    _ = MiMModule(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode="mod",
        inter_depth=0,
        model="mixmae",
        model_size="tiny",
    )


@pytest.mark.xfail
def test_model_group_mixmae_should_fail(dataset_treesat, mask):
    _ = MiMModule(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode="group",
        inter_depth=0,
        model="mixmae",
        model_size="tiny",
    )


@pytest.mark.xfail
def test_levels_should_fail(dataset_treesat, mask):
    _ = MiMModule(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode="shared",
        inter_depth=0,
        model="mixmae",
        model_size="tiny",
        num_levels=1,
    )


@pytest.mark.xfail
def test_model_should_fail(dataset_treesat, mask):
    _ = MiMModule(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode="shared",
        inter_depth=0,
        model="MAE",
        model_size="tiny",
    )


@pytest.mark.xfail
def test_missing_args_should_fail():
    _ = MiMModule(model="mae", model_size="tiny")
