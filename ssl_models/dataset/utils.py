"""Utils module."""

from datetime import datetime

import numpy as np


def dates_numpy(dates: list[datetime]) -> np.ndarray:
    """Convert list of dates to (year, day of year, hour) numpy array."""
    return np.array(
        [[date.year, date.timetuple().tm_yday, date.hour] for date in dates],
        dtype=np.int16,
    )


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
    datetimes = datetime.strptime(  # noqa: DTZ007
        datetime_str.split("_")[-1][:8],
        "%Y%m%d",
    )

    return np.array(
        np.datetime64(datetimes).astype(np.int64),
        ndmin=1,
    )


def dict_datetimes(datetime_dict: dict) -> np.ndarray:
    """Convert list of datetime strings to list of datetimes."""
    datetimes = [
        datetime.strptime(  # noqa: DTZ007
            str(datetime_dict[str(idx)]),
            "%Y%m%d",
        )
        for idx in range(len(datetime_dict))
    ]
    return dates_numpy(datetimes)
