"""Utils module."""

import datetime as dt
from typing import Literal

import numpy as np
import pandas as pd
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

def filter_masks(
    masks: np.ndarray,
    mode: Literal[0, 1, 2] = 2,  # 0 = snow, 1 = clouds, 2 = both
    area_threshold: float = 0.5,
    proba_threshold: int = 60,
) -> list[int]:
    """Remove images with excessive cloud or snow cover from the time series."""
    index_to_keep = []
    for t in range(masks.shape[0]):
        if mode != 2:  # noqa: PLR2004
            cover = np.count_nonzero(masks[t, mode, :, :] >= proba_threshold)
        else:
            cover = np.count_nonzero(
                masks[t, 0, :, :] >= proba_threshold,
            ) + np.count_nonzero(masks[t, 1, :, :] >= proba_threshold)
        cover = cover / (masks.shape[2] * masks.shape[3])
        if cover < area_threshold:
            index_to_keep.append(t)

    return index_to_keep


def get_day_number_in_year(date_arr: Tensor, place: int = 2) -> Tensor:
    """Convert date to day of year."""
    day_number = []
    for date_string in date_arr:
        date_object = dt.datetime.strptime(
            str(date_string).split("_")[place][:8],
            "%Y%m%d",
        ).replace(tzinfo=dt.timezone.utc)
        day_number.append(date_object.timetuple().tm_yday)
    return day_number


def average_monthly_image(
    data: Tensor,
    dates: list[dt.datetime],
    diff_dates: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Average monthly time series images."""
    average_data, average_dates = [], []
    month_range = pd.period_range(
        start=dates[0].strftime("%Y-%m-%d"),
        end=dates[-1].strftime("%Y-%m-%d"),
        freq="M",
    )
    for m in month_range:
        month_dates = list(
            filter(
                lambda i: (dates[i].month == m.month) and (dates[i].year == m.year),
                range(len(dates)),
            ),
        )
        if month_dates:
            average_data.append(np.mean(np.take(data, month_dates), axis=0))
            average_dates.append(np.mean(np.take(diff_dates, month_dates)))
    return np.array(average_data), np.array(average_dates)


def pad_tensor(x: Tensor, length: int, pad_value: int = 0) -> Tensor:
    """Generate dummy satellite images for missing dates."""
    padlen = length - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_satellite_data(
    sat_data: Tensor,
    dates: Tensor,
    pad_value: int = 0,
    ts_size: int = 12,
) -> tuple[Tensor, Tensor]:
    """Pad tensor iteratively on list of satellite image series."""
    sizes = sat_data.shape[0]
    max_size = ts_size
    padded_data, padded_dates = [], []
    if sizes != max_size:
        padded_data = pad_tensor(sat_data, max_size, pad_value=pad_value)
        padded_dates = pad_tensor(dates, max_size, pad_value=pad_value)
    else:
        padded_data = sat_data
        padded_dates = dates

    return padded_data, padded_dates


def clip_ts(
    data: np.ndarray,
    diff_dates: list[float],
    clip_ts_val: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip the time series images."""
    rng = np.random.default_rng()
    random_indices = rng.permutation(len(data))[:clip_ts_val]
    return np.take(data, random_indices, axis=0), np.take(diff_dates, random_indices)