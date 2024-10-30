"""Utils module.

Code adapted from:
    - https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    - https://github.com/Sense-X/MixMIM/blob/master/util/pos_embed.py
"""

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
        msg = f"Invalid pos embedding dim {dim}. Expected multiple of 4"
        raise ValueError(msg)
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y[:, :, None] * omega[None, None, :]
    x = x[:, :, None] * omega[None, None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)
    return pe.to(dtype)


def downsampling_from_patch_size(unpool: bool, patch_size: int) -> int:
    """Infer number of unpooling downsamplings associated with a given patch size."""
    if unpool:
        num_downsampling = round(np.log2(patch_size))
        if 2**num_downsampling != patch_size:
            msg = (
                f"Invalid patch size {patch_size}."
                " Expected power of 2 when using unpooling."
            )
            raise ValueError(msg)
    else:
        num_downsampling = None
    return num_downsampling
