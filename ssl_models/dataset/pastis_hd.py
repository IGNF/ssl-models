"""PASTIS-HD dataset module."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ssl_models.dataset.utils import dict_datetimes


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
            dict_datetimes(meta_data.loc[image_id, "dates-S1A"])
            for image_id in image_ids
        ]
        self.s1_asc_dates = [
            dict_datetimes(meta_data.loc[image_id, "dates-S1A"])
            for image_id in image_ids
        ]
        self.s1_des_dates = [
            dict_datetimes(meta_data.loc[image_id, "dates-S1A"])
            for image_id in image_ids
        ]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        image_id = self.image_ids[idx]
        spot_path = (
            self.root_dir
            / "DATA_SPOT"
            / "PASTIS_SPOT6_RVB_1M00_2019"
            / f"SPOT6_RVB_1M00_2019_{image_id}.tif"
        )

        meta = {}
        meta["spot_path"] = spot_path
        meta["spot_dates"] = np.array(
            np.datetime64("2019-07-01").astype(np.int64),
            ndmin=1,
        )

        meta["s2_path"] = self.root_dir / "DATA_S2" / f"S2_{image_id}.tif"
        meta["s2_dates"] = self.s2_dates[idx]
        meta["s1_asc_path"] = self.root_dir / "DATA_S1A" / f"S1A_{image_id}.npy"
        meta["s1_asc_dates"] = self.s1_asc_dates[idx]
        meta["s1_des_path"] = self.root_dir / "DATA_S1D" / f"S1D_{image_id}.npy"
        meta["s1_des_dates"] = self.s1_des_dates[idx]

        meta["pastis_seg_path"] = (
            self.root_dir / "ANNOTATIONS" / f"TARGET_{image_id}.npy"
        )
        meta["pastis_seg_dates"] = np.array(
            np.datetime64("2019-07-01").astype(np.int64),
            ndmin=1,
        )

        inputs = self.preprocess_rasters(meta)

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.aerial_paths)
