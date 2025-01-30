"""S2-NAIP dataset module."""

from pathlib import Path
from typing import Literal

import numpy as np

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import naip_datetimes, products_datetimes, read_csv


class S2NAIPDataset(GenericDataset):
    """S2-NAIP dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_dates: bool,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        random_crop: bool,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )

        csv_data = read_csv(
            csv_dir=root_dir,
            stage=stage,
            ssl_phase=ssl_phase,
        )

        self.root_dir = root_dir
        self.image_ids = csv_data["name"].to_list()
        self.base_length = len(self.image_ids)
        self.repeats = dataset.repeats

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        idx, start_gcd = self.sample_gcd(
            idx,
            base_length=self.base_length,
            repeats=self.repeats,
        )

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

        inputs = self.preprocess_rasters(meta, start_gcd=start_gcd)

        inputs["ref_date"] = inputs["naip_dates"]

        return self.transform_rasters(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.base_length * self.repeats**2
