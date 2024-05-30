"""
Code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
Code from: https://github.com/Sense-X/MixMIM/blob/master/util/pos_embed.py
"""
import numpy as np
import torch


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(length, dim, temperature: int = 10000, dtype=torch.float32):
    assert (
        dim % 2 == 0
    ), "Feature dimension must be a multiple of 2 for sincos embedding"
    position = torch.arange(length)

    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_pos = position[:, None] * omega[None, :]
    pe = torch.cat((scaled_pos.sin(), scaled_pos.cos()), dim=1)

    return pe.type(dtype)

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/legacy/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, cossin=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, cossin=cossin)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, cossin=False):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], cossin=cossin)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], cossin=cossin)  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, cossin=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    if cossin:
        emb = np.concatenate([emb_cos, emb_sin], axis=1)  # (M, D)
    else:
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb