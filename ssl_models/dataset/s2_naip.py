"""S2-NAIP dataset module."""

from pathlib import Path

import numpy as np
import pandas as pd

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import naip_datetimes, products_datetimes


class S2NAIPDataset(GenericDataset):
    """S2-NAIP dataset."""

    def __init__(
        self,
        root_dir: Path,
        csv_path: Path,
        use_transform: bool,
        dataset: DatasetConfig,
        *args,  # noqa: ARG002, ANN002
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            use_transform=use_transform,
            dataset=dataset,
        )

        csv_data = pd.read_csv(csv_path)

        self.root_dir = root_dir
        self.image_ids = csv_data["name"].to_list()

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        image_id = self.image_ids[idx]

        meta = {}
        meta["naip_path"] = self.root_dir / "naip" / f"{image_id}.png"
        meta["naip_dates"] = naip_datetimes(
            str(
                np.loadtxt(
                    self.root_dir / "dates" / "naip" / f"{image_id}.txt",
                    dtype="str",
                ),
            ),
        )

        landsat_resolutions = {"10": "8", "20": "16"}
        for res, suffix in landsat_resolutions.items():
            meta[f"landsat_{res}m_path"] = (
                self.root_dir / "landsat" / f"{image_id}_{suffix}.tif"
            )
            meta[f"landsat_{res}m_dates"] = products_datetimes(
                np.loadtxt(
                    self.root_dir / "dates" / "landsat" / f"{image_id}.txt",
                    dtype="str",
                ),
                4,
            )

        s2_resolutions = {"10": "8", "20": "16", "40": "32"}
        for res, suffix in s2_resolutions.items():
            meta[f"s2_{res}m_path"] = (
                self.root_dir / "sentinel2" / f"{image_id}_{suffix}.tif"
            )
            meta[f"s2_{res}m_dates"] = products_datetimes(
                np.loadtxt(
                    self.root_dir / "dates" / "s2" / f"{image_id}.txt",
                    dtype="str",
                ),
                5,
            )

        meta["s1_path"] = self.root_dir / "sentinel1" / f"{image_id}.tif"
        meta["s1_dates"] = products_datetimes(
            np.loadtxt(self.root_dir / "dates" / "s1" / f"{image_id}.txt", dtype="str"),
            5,
        )

        meta["osm_seg_path"] = self.root_dir / "openstreetmap" / f"{image_id}.geojson"
        meta["osm_seg_dates"] = meta["naip_dates"]

        inputs = self.preprocess_rasters(meta)

        inputs["ref_date"] = inputs["naip_dates"]

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.image_ids)
