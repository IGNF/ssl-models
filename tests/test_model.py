import itertools

import pytest

from ssl_models.conf.loss import MAELossConfig
from ssl_models.train.lit_mae import LitMAE


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
    ssl_module = LitMAE(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
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
    ssl_module = LitMAE(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
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
    ssl_module = LitMAE(
        dataset=dataset_treesat,
        mask=mask,
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        model_size=model_size,
        type_head=type_head,
        interpolate=interpolate,
        loss=MAELossConfig(name=loss),
    )
    ssl_module.setup(stage=stage)


@pytest.mark.xfail
def test_missing_args_should_fail():
    _ = LitMAE(model_size="tiny")
