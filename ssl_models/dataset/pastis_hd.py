"""PASTIS-HD dataset module."""

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from ssl_models.dataset.utils import dict_to_datetime


class PASTISHDDataset(Dataset):
    """PASTIS-HD dataset."""

    def __init__(
        self,
        root_dir: str | Path,
        csv_path: str | Path,
    ) -> None:
        image_ids = pd.read_csv(csv_path)["id"]
        meta_data = gpd.read_file(root_dir / "metadata.geojson").set_index("id")

        self.root_dir = Path(root_dir)
        self.image_ids = image_ids

        self.s2_names = [f"S2_{image_id}" for image_id in image_ids]
        self.s1_asc_names = [f"S1A_{image_id}" for image_id in image_ids]
        self.s1_des_names = [f"S1D_{image_id}" for image_id in image_ids]
        self.spot_names = [f"SPOT6_RVB_1M00_2019_{image_id}" for image_id in image_ids]

        self.s2_dates = [
            dict_to_datetime(meta_data.loc[image_id, "dates-S1A"])
            for image_id in image_ids
        ]
        self.s1_asc_dates = [
            dict_to_datetime(meta_data.loc[image_id, "dates-S1A"])
            for image_id in image_ids
        ]
        self.s1_des_dates = [
            dict_to_datetime(meta_data.loc[image_id, "dates-S1A"])
            for image_id in image_ids
        ]

    def __getitem__(self, idx: int) -> dict[str, Tensor | list]:
        """Get dataset item."""
        image_id = self.image_ids[idx]
        row, col = self.get_row_col()

        meta = {}

        spot_path = (
            self.root_dir
            / "DATA_SPOT"
            / "PASTIS_SPOT6_RVB_1M00_2019"
            / f"SPOT6_RVB_1M00_2019_{image_id}.tif"
        )
        meta["spot_path"] = spot_path
        meta["spot_dates"] = [datetime(2019, 6, 1)]  # noqa: DTZ001

        meta["s2_path"] = self.root_dir / "DATA_S2" / f"S2_{image_id}.tif"
        meta["s2_dates"] = self.s2_dates[idx]

        meta["s1_asc_path"] = self.root_dir / "DATA_S1A" / f"S1A_{image_id}.npy"
        meta["s1_asc_dates"] = self.s1_asc_dates[idx]

        meta["s1_des_path"] = self.root_dir / "DATA_S1D" / f"S1D_{image_id}.npy"
        meta["s1_des_dates"] = self.s1_des_dates[idx]

        inputs = super().preprocess_rasters(meta)
        return self.transform(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.aerial_paths)
