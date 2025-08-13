"""Utils module."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

from ssl_models import RNG


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


def cloud_mae_masking(
    cloud_mask: np.ndarray,
    cloud_threshold: float,
    # mask_ratio: float,
    mask_value: float,
    mae_masking_ratio: float = 0.1,
) -> np.ndarray:
    """Add random masked pixels to cloud mask."""

    D, C, H, W = cloud_mask.shape  # noqa: N806
    # cloud_mask = cloud_mask_correction(cloud_mask)  # TODO à mettre dans dataset avant filtrage date

    cloud_mask = np.where(cloud_mask <= cloud_threshold, 1, 0)
    cloud_mask = cloud_mask.flatten()
    nb_visible = np.count_nonzero(cloud_mask == 1)
    visible_indices = np.where(cloud_mask == 1)[0]
    num_mae_masked = round(nb_visible * mae_masking_ratio)
    if num_mae_masked > 0:
        mae_masked_indices = RNG.choice(
            visible_indices,
            size=num_mae_masked,
            replace=False,
        )
        cloud_mask[mae_masked_indices] = mask_value

    return rearrange(
        cloud_mask,
        "(d c h w) -> d c h w",
        d=D,
        c=C,
        h=H,
        w=W,
    )


def cloud_mae_masking_test(
    cloud_mask: np.ndarray,
    cloud_threshold: float,
    mask_value: float,
) -> np.ndarray:
    """Process test cloud mask."""
    cloud_mask = np.where(cloud_mask >= mask_value, mask_value, cloud_mask)
    return np.where(
        (cloud_mask <= cloud_threshold) & (cloud_mask != mask_value),
        1,
        0,
    )


def mask_local_global_consistency(
    dates_local: np.ndarray,
    mask_local: np.ndarray,
    dates_global: np.ndarray,
    mask_global: np.ndarray,
    mask_value: float,
    cloud_threshold: float,
) -> np.ndarray:
    """Mask the global mask with the pixels masked in the local mask."""
    mask_global = np.where(mask_global >= mask_value, 0, mask_global)
    mask_global = np.where(mask_global <= cloud_threshold, 1, 0)

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

    # common dates
    _, common_local_indices, common_global_indices = np.intersect1d(
        dates_local,
        dates_global,
        return_indices=True,
    )

    mask_global[common_global_indices, ...] = mask_local[
        common_local_indices,
        ...,
    ]

    return np.where(mask_global >= mask_value, 0, mask_global)


def pad_dates_modalities(batch: list[dict], pad_dim=0):
    max_dates = max(t["s2_zoom"].shape[0] for t in batch)
    min_dates = min(t["s2_zoom"].shape[0] for t in batch)

    modalities = list(batch[0].keys())

    modalities.remove("ref_date")
    pad_value = 0  # vérifier que les vrais nuages sont maskés par des 0  TODO

    if min_dates == max_dates:
        return {
            key: torch.stack([torch.from_numpy(item[key]) for item in batch])
            for key in batch[0]
        }
    else:
        padded_batch = {}
        for name_mod in modalities:
            padded_batch[name_mod] = torch.from_numpy(
                np.stack(
                    [
                        np.pad(
                            b[name_mod],
                            [
                                (max_dates - b[name_mod].shape[pad_dim], 0)
                                if i == pad_dim
                                else (0, 0)
                                for i in range(b[name_mod].ndim)
                            ],
                            constant_values=pad_value,
                        )
                        for b in batch
                    ]
                )
            )
        padded_batch["ref_date"] = torch.from_numpy(
            np.stack([b["ref_date"] for b in batch])
        )
        return padded_batch
