import itertools

import pytest

from ssl_models.ssl.mixmae import (
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
    "model_size, fusion_mode",
    itertools.product(
        ("tiny",),
        ("msgfm", "shared"),
    ),
)
def test_mixmae(
    datasets_treesat,
    mask,
    model_map,
    model_size,
    fusion_mode,
):
    model_args = {
        "datasets": datasets_treesat,
        "mask": mask,
        "fusion_mode": fusion_mode,
        "model": "mixmae",
        "num_levels": 3,
        "swin_version": 1,
        "v2_attn": True,
        "type_head": "attentive",
        "interpolate": "nearest",
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map[model_size](**model_args)


@pytest.mark.parametrize(
    "model_size, interpolate, num_levels, swin_version, v2_attn, type_head",
    itertools.product(
        ("tiny",),
        ("nearest", "bilinear"),
        (3, 4),
        (1, 2),
        (True, False),
        ("attentive", "linear"),
    ),
)
def test_mixmae_arch(
    datasets_treesat,
    mask,
    model_map,
    model_size,
    interpolate,
    num_levels,
    swin_version,
    v2_attn,
    type_head,
):
    model_args = {
        "datasets": datasets_treesat,
        "mask": mask,
        "fusion_mode": "shared",
        "model": "mixmae",
        "num_levels": num_levels,
        "swin_version": swin_version,
        "v2_attn": v2_attn,
        "type_head": type_head,
        "interpolate": interpolate,
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map[model_size](**model_args)


@pytest.mark.xfail
def test_mod_mixmae_should_fail(datasets_treesat, mask):
    model_args = {
        "datasets": datasets_treesat,
        "mask": mask,
        "fusion_mode": "mod",
        "model": "mixmae",
        "num_levels": 3,
        "swin_version": 1,
        "v2_attn": True,
        "type_head": "attentive",
        "interpolate": "nearest",
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map["tiny"](**model_args)


@pytest.mark.xfail
def test_group_mixmae_should_fail(datasets_treesat, mask):
    model_args = {
        "datasets": datasets_treesat,
        "mask": mask,
        "fusion_mode": "group",
        "model": "mixmae",
        "num_levels": 3,
        "swin_version": 1,
        "v2_attn": True,
        "type_head": "attentive",
        "interpolate": "nearest",
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map["tiny"](**model_args)
