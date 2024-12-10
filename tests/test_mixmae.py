import itertools

import pytest
import torch

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
    "model_size, multimodal",
    itertools.product(
        ("tiny",),
        ("msgfm", "shared"),
    ),
)
def test_mixmae(
    datasets_geopile,
    mask,
    model_map,
    model_size,
    multimodal,
):
    model_args = {
        "datasets": datasets_geopile,
        "mask": mask,
        "multimodal": multimodal,
        "model": "mixmae",
        "unpool": False,
        "num_levels": 3,
        "swin_version": 1,
        "v2_attn": True,
        "type_head": "attentive",
        "loss_fn": torch.square,
        "norm_pix_loss": True,
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map[model_size](**model_args)


@pytest.mark.parametrize(
    "model_size, loss_fn, norm_pix_loss",
    itertools.product(
        ("tiny",),
        (torch.square, torch.abs),
        (True, False),
    ),
)
def test_mixmae_loss(
    datasets_geopile,
    mask,
    model_map,
    model_size,
    loss_fn,
    norm_pix_loss,
):
    model_args = {
        "datasets": datasets_geopile,
        "mask": mask,
        "multimodal": "shared",
        "model": "mixmae",
        "unpool": False,
        "num_levels": 3,
        "swin_version": 1,
        "v2_attn": True,
        "type_head": "attentive",
        "loss_fn": loss_fn,
        "norm_pix_loss": norm_pix_loss,
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map[model_size](**model_args)


@pytest.mark.parametrize(
    "model_size, num_levels, swin_version, v2_attn, type_head",
    itertools.product(
        ("tiny",),
        (3, 4),
        (1, 2),
        (True, False),
        ("attentive", "linear"),
    ),
)
def test_mixmae_arch(
    datasets_geopile,
    mask,
    model_map,
    model_size,
    num_levels,
    swin_version,
    v2_attn,
    type_head,
):
    model_args = {
        "datasets": datasets_geopile,
        "mask": mask,
        "multimodal": "shared",
        "model": "mixmae",
        "unpool": False,
        "num_levels": num_levels,
        "swin_version": swin_version,
        "v2_attn": v2_attn,
        "type_head": type_head,
        "loss_fn": torch.square,
        "norm_pix_loss": True,
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map[model_size](**model_args)


@pytest.mark.xfail
def test_mod_mixmae_should_fail(datasets_geopile, mask):
    model_args = {
        "datasets": datasets_geopile,
        "mask": mask,
        "multimodal": "mod",
        "model": "mixmae",
        "unpool": False,
        "num_levels": 3,
        "swin_version": 1,
        "v2_attn": True,
        "type_head": "attentive",
        "loss_fn": torch.square,
        "norm_pix_loss": True,
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map["tiny"](**model_args)


@pytest.mark.xfail
def test_group_mixmae_should_fail(datasets_geopile, mask):
    model_args = {
        "datasets": datasets_geopile,
        "mask": mask,
        "multimodal": "group",
        "model": "mixmae",
        "unpool": False,
        "num_levels": 3,
        "swin_version": 1,
        "v2_attn": True,
        "type_head": "attentive",
        "loss_fn": torch.square,
        "norm_pix_loss": True,
        "fac_abs_enc": 1.0,
        "fac_date_enc": 1.0,
    }
    model_map["tiny"](**model_args)
