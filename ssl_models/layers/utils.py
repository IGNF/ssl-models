"""Utils module."""

import itertools

import numpy as np
import torch
from torch import Tensor

from ssl_models import RNG


def shuffle_enc_to_dec(x: dict[str, Tensor]) -> dict[str, Tensor]:
    """Shuffle modalities and dates."""
    name_ids = []
    num_patches = []
    for name_mod in x:
        for idx_date in range(x[name_mod].shape[1]):
            name_ids.append((name_mod, idx_date))
            num_patches.append(x[name_mod].shape[2])
    compat_mat = np.array(num_patches)[:, None] == np.array(num_patches)[None, :]
    compat_mat = compat_mat.astype(np.float32) - np.eye(
        len(num_patches),
        dtype=np.float32,
    )

    x_dec = {name_mod: x[name_mod].clone() for name_mod in x}
    for name_out in x_dec:
        for idx_batch, idx_out in itertools.product(
            range(x_dec[name_out].shape[0]),
            range(x_dec[name_out].shape[1]),
        ):
            compat_row = compat_mat[name_ids.index((name_out, idx_out))]
            if RNG.choice([True, False]) and compat_row.sum() > 0:
                idx_in = RNG.choice(
                    range(len(name_ids)),
                    p=compat_row / compat_row.sum(),
                )
                (name_in, idx_in) = name_ids[idx_in]
                x_dec[name_out][idx_batch, idx_out] = x[name_in][idx_batch, idx_in]
    return x_dec


def flatten_dates(
    tensor_dict: dict[str, Tensor],
    dim: int,
) -> dict[str, Tensor]:
    """Flatten date dimension in tensor dict."""
    for name_mod in tensor_dict:
        tensor_dict[name_mod] = tensor_dict[name_mod].flatten(dim, dim + 1)
    return tensor_dict


def unflatten_dates(
    tensor_dict: dict[str, Tensor],
    dim: int,
    num_dates: dict[str, int],
) -> dict[str, Tensor]:
    """Unflatten date dimension in tensor dict."""
    for name_mod in tensor_dict:
        if dim == 0:
            tensor_dict[name_mod] = tensor_dict[name_mod].unflatten(
                0,
                (-1, num_dates[name_mod]),
            )
        elif dim == 1:
            tensor_dict[name_mod] = tensor_dict[name_mod].unflatten(
                1,
                (num_dates[name_mod], -1),
            )
        else:
            msg = f"Invalid dimension {dim} to unflatten"
            raise ValueError(msg)
    return tensor_dict


def reshape_encoding(encoding: Tensor, grid_size: int) -> Tensor:
    """Reshape encoding for a modality, based on grid factor."""
    encoding = encoding.reshape((1,) * (5 - encoding.ndim) + encoding.shape)
    encoding = encoding.expand(
        (
            *encoding.shape[:-3],
            max(grid_size, encoding.shape[-3]),  # max is for num_dates=1
            max(grid_size, encoding.shape[-2]),
            encoding.shape[-1],
        ),
        # N.B.: expand returns just another view of data and does not clone it,
        # so inplace operations should not be performed after it
    )
    encoding = encoding.unflatten(-3, (grid_size, -1))
    encoding = encoding.unflatten(-2, (grid_size, -1))
    return encoding.mean(dim=(-2, -4)).flatten(-3, -2)


def encode_dates(
    dates: Tensor,
    ref_date: Tensor,
    dim: int,
    date_dim: int,
    fac_date_enc: float,
    grid_size: int,
) -> Tensor:
    """Encode day of year, hour of day, and difference to reference date."""
    dates = dates.reshape(dates.shape + (1,) * (5 - dates.ndim))
    ref_date = ref_date.reshape(ref_date.shape + (1,) * (5 - ref_date.ndim))

    year = dates[:, :, 0]
    doy = dates[:, :, 1] / 365.25
    hour = dates[:, :, 2] / 24.0

    year_ref = ref_date[:, :, 0]
    doy_ref = ref_date[:, :, 1] / 365.25
    diff = (year + doy) - (year_ref + doy_ref)

    doy = 2 * np.pi * doy
    hour = 2 * np.pi * hour
    date_encoding = torch.stack(
        [
            diff,
            torch.sin(doy),
            torch.cos(doy),
            torch.sin(hour),
            torch.cos(hour),
        ],
        dim=-1,
    )
    date_encoding = reshape_encoding(date_encoding, grid_size=grid_size)
    date_encoding *= fac_date_enc

    diff, date_encoding = torch.split(date_encoding, [1, 4], dim=-1)
    pad_zeros = torch.zeros_like(diff).expand(-1, -1, -1, dim - date_dim)
    pad_diff = diff.expand(-1, -1, -1, date_dim - 4)
    return torch.cat([pad_zeros, pad_diff, date_encoding], dim=-1)


def posemb_sincos_2d(
    h: int,
    w: int,
    dim: int,
    date_dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create 2d positional encodings."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    if dim % 4 or date_dim % 4:
        msg = f"Invalid embedding dimensions {dim}, {date_dim}. Expected multiples of 4"
        raise ValueError(msg)
    omega = torch.arange((dim - date_dim) // 4) / ((dim - date_dim) // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y[:, :, None] * omega[None, None, :]
    x = x[:, :, None] * omega[None, None, :]
    pos_encoding = torch.cat(
        [x.sin(), x.cos(), y.sin(), y.cos(), torch.zeros((h, w, date_dim))],
        dim=-1,
    )
    return pos_encoding.to(dtype)


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
