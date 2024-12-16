"""Utils module."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def dates_numpy(dates: list[datetime]) -> np.ndarray:
    """Convert list of dates to (year, day of year, hour) numpy array."""
    return np.array(
        [[date.year, date.timetuple().tm_yday, date.hour] for date in dates],
        dtype=np.int16,
    )


def strs_datetimes(date_strs: list[str], format_: str = "%Y-%m-%d") -> np.ndarray:
    """Convert list of strs to list of datetimes."""
    datetimes = [datetime.strptime(date_str, format_) for date_str in date_strs]  # noqa: DTZ007
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
    root_dir: Path,
    stage: Literal["train", "val", "test"],
    ssl_phase: Literal["pretrain", "probe", "finetune"],
    balance_pretrain: bool,
    **kwargs,  # noqa: ANN003
) -> Path:
    """Read dataset csv based on SSL phase and choice of balancing."""
    match ssl_phase:
        case "pretrain":
            csv_path = (
                root_dir / f"{stage}_balanced.csv"
                if balance_pretrain
                else root_dir / f"{stage}.csv"
            )
        case "probe" | "finetune":
            csv_path = root_dir / f"{stage}.csv"
    return pd.read_csv(csv_path, **kwargs)
