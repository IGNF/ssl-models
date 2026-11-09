import itertools

import pytest

from ssl_models.mim.mae import mae_large, mae_medium, mae_small, mae_tiny


@pytest.fixture()
def model_map():
    return {
        "tiny": mae_tiny,
        "small": mae_small,
        "medium": mae_medium,
        "large": mae_large,
    }


@pytest.mark.parametrize(
    "model_size, fusion_mode",
    itertools.product(
        ("tiny",),
        ("msgfm", "shared", "monotemp", "mod", "group"),
    ),
)
def test_mae(datasets_treesat, mask, model_map, model_size, fusion_mode):
    model_args = {
        "datasets": datasets_treesat,
        "mask": mask,
        "fusion_mode": fusion_mode,
        "model": "mae",
        "inter_depth": 0,
        "num_levels": 1,
        "type_head": "attentive",
        "interpolate": "nearest",
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map[model_size](**model_args)


@pytest.mark.parametrize(
    "model_size, type_head, interpolate, fac_abs_enc, fac_date_enc",
    itertools.product(
        ("tiny",),
        ("attentive", "linear"),
        ("nearest", "bilinear"),
        (1.0, 0.0),
        (1.0, 0.0),
    ),
)
def test_mae_arch(
    datasets_treesat,
    mask,
    model_map,
    model_size,
    type_head,
    interpolate,
    fac_abs_enc,
    fac_date_enc,
):
    model_args = {
        "datasets": datasets_treesat,
        "mask": mask,
        "fusion_mode": "mod",
        "model": "mae",
        "inter_depth": 0,
        "num_levels": 1,
        "type_head": type_head,
        "interpolate": interpolate,
        "fac_abs_enc": fac_abs_enc,
        "fac_date_enc": fac_date_enc,
    }
    model_map[model_size](**model_args)
