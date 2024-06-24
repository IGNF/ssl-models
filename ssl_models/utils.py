"""Pos encoding module. Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py."""

import numpy as np
import torch
from torch import Tensor


def posemb_sincos_2d(
    h: int,
    w: int,
    dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create 2d positional encodings."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    if dim % 4:
        raise ValueError
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.to(dtype)


def posemb_sincos_1d(
    length: int,
    dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create 1d positional encodings."""
    if dim % 2:
        raise ValueError
    position = torch.arange(length)

    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_pos = position[:, None] * omega[None, :]
    pe = torch.cat((scaled_pos.sin(), scaled_pos.cos()), dim=1)

    return pe.to(dtype)
