"""Utils module."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from einops import rearrange


def dates_numpy(dates: list[datetime]) -> np.ndarray:
    """Convert list of dates to (year, day of year, hour) numpy array."""
    return np.array(
        [[date.year, date.timetuple().tm_yday, date.hour] for date in dates],
        dtype=np.int16,
    )


def strs_datetimes(date_strs: list[str], fmt: str = "%Y-%m-%d") -> np.ndarray:
    """Convert list of strs to list of datetimes."""
    date_strs = [
        date_str[:-2] + "01" if date_str[-2:] == "00" else date_str
        for date_str in date_strs
    ]
    datetimes = [datetime.strptime(date_str, fmt) for date_str in date_strs]  # noqa: DTZ007
    return dates_numpy(datetimes)


def products_datetimes(products: list[str] | list[bytes], idx: int) -> np.ndarray:
    """Convert list of products to list of datetimes."""
    if isinstance(products[0], str):
        datetimes = [
            datetime.strptime(  # noqa: DTZ007
                product.split("_")[-idx][:8],
                "%Y%m%d",
            )
            for product in products
        ]
    else:
        datetimes = [
            datetime.strptime(  # noqa: DTZ007
                product.decode().split("_")[-idx][:8],
                "%Y%m%d",
            )
            for product in products
        ]
    return dates_numpy(datetimes)


def naip_datetimes(datetime_str: str) -> np.ndarray:
    """Convert naip string to list of datetime."""
    datetimes = [
        datetime.strptime(  # noqa: DTZ007
            datetime_str.split("_")[-1][:8],
            "%Y%m%d",
        ),
    ]
    return dates_numpy(datetimes)


def dict_datetimes(datetime_dict: dict, start: int = 0) -> np.ndarray:
    """Convert list of datetime strings to list of datetimes."""
    datetimes = [
        datetime.strptime(  # noqa: DTZ007
            str(datetime_dict[str(idx)]),
            "%Y%m%d",
        )
        for idx in range(start, len(datetime_dict) + start)
    ]
    return dates_numpy(datetimes)


def read_csv(
    csv_dir: Path,
    stage: Literal["train", "val", "test"],
    ssl_phase: Literal["pretrain", "probe", "finetune"],
    version: str | None = None,
    balance_pretrain: bool = False,
    val_pretrain: bool = False,
    filter_percent: int | None = None,
    fold: int | None = None,
    **kwargs,  # noqa: ANN003
) -> Path:
    """Read dataset csv."""
    csv_name = []
    if version:
        csv_name += [version]
    if balance_pretrain and ssl_phase == "pretrain":
        csv_name += ["balanced"]
    if filter_percent:
        csv_name += [f"filtered_{filter_percent}"]
    if fold:
        csv_name += [f"fold_{fold}"]

    stages = (
        ["train", "val"]
        if stage == "train" and ssl_phase == "pretrain" and val_pretrain
        else [stage]
    )
    return pd.concat(
        [
            pd.read_csv(csv_dir / f"{'_'.join([stage, *csv_name])}.csv", **kwargs)
            for stage in stages
        ],
    )


def cloud_mask_correction(input_mask: np.ndarray, threshold: int = 50) -> np.ndarray:
    """Correct cloud masks by identifying and removing inconsistent pixels based on temporal statistics."""  # noqa: E501

    def compute_persistence_mask(array: np.ndarray) -> np.ndarray:
        """Binarize the input array (non-zero values become 1) and sums along the time axis."""  # noqa: E501
        binary_array = (array != 0).astype(int)  # Binarization
        return binary_array.sum(axis=0)

    cloud_masks = input_mask.copy()
    persistence_mask = compute_persistence_mask(cloud_masks)  # Temporal stationarity
    pixel_threshold = np.percentile(persistence_mask, threshold)
    error_indexes = np.where(persistence_mask > pixel_threshold)
    cloud_masks[:, error_indexes[0], error_indexes[1]] = 0
    return np.stack(cloud_masks, axis=0)


def custom_mae_masking(
    cloud_mask: np.ndarray,
    cloud_threshold: float,
    mask_ratio: float,
    mask_value: float,
) -> np.ndarray:
    """Add random masked pixels to cloud mask."""
    D, C, H, W = cloud_mask.shape  # noqa: N806
    L = H * W  # noqa: N806
    cloud_mask = cloud_mask_correction(cloud_mask)
    cloud_mask = rearrange(cloud_mask, "d c h w -> d (h w c)")

    cloud_mask = (cloud_mask <= cloud_threshold).astype(np.float16)  # 0: cloud
    num_masked = (cloud_mask == 0).sum(axis=-1)
    num_add_mask = np.clip(round(mask_ratio * L) - num_masked, 0, L)

    rng = np.random.default_rng(42)
    rand_vals = rng.random((D, L))
    threshold_vals = np.take_along_axis(
        np.sort(rand_vals, axis=-1),
        num_add_mask[..., None] - 1,
        axis=-1,
    )
    random_mask = (rand_vals <= threshold_vals).astype(np.float16)  # 1: mask

    cloud_mask[cloud_mask == 1] = (random_mask[cloud_mask == 1] * mask_value) + (
        1 - random_mask[cloud_mask == 1]
    )

    return rearrange(cloud_mask, "d (h w c) -> d c h w", d=D, c=C, h=H, w=W)


def mask_local_global_consistency(
    dates_local: np.ndarray,
    mask_local: np.ndarray,
    dates_global: np.ndarray,
    mask_global: np.ndarray,
    mask_value: float,
) -> np.ndarray:
    """Mask the global mask with the pixels masked in the local mask."""
    _, C, H, W = mask_global.shape  # noqa: N806
    dates_local = dates_local.reshape(-1, 3)
    dates_global = dates_global.reshape(-1, 3)

    dtype = np.dtype(
        [
            ("f0", dates_local.dtype),
            ("f1", dates_local.dtype),
            ("f2", dates_local.dtype),
        ],
    )
    dates_local = dates_local.view(dtype)
    dates_global = dates_global.view(dtype)

    # finding common dates
    _, common_local_indices, common_global_indices = np.intersect1d(
        dates_local,
        dates_global,
        return_indices=True,
    )
    mask_global[common_global_indices, ...] = mask_local[
        common_local_indices,
        ...,
    ]
    # not needed for reconstruction
    mask_global = np.where(mask_global == 0, mask_value, mask_global)

    uncommon_global_indices = np.setdiff1d(
        np.arange(mask_global.shape[0]),
        common_global_indices,
    )
    mask_global[uncommon_global_indices, ...] = np.ones(
        (len(uncommon_global_indices), C, H, W),
    )
    return mask_global
