"""TreeSatAI-TS dataset."""

from pathlib import Path

import albumentations as A  # noqa: N812
import h5py
import numpy as np
import pandas as pd
import rasterio

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import products_to_datetimes


class TreeSatAITSDataset(GenericDataset):
    """TreeSatAI-TS dataset."""

    MOD_MAPPING: dict[str, str] = {  # noqa: RUF012
        "sen-2": "s2",
        "sen-1-asc": "s1_asc",
        "sen-1-des": "s1_des",
    }
    SHIFT_AERIAL = (2, 2)  # aerial images have buffers of 2 pixels on each side

    def __init__(
        self,
        root_dir: Path,
        csv_path: Path,
        transform: A.Compose | None,
        dataset: DatasetConfig,
        *args,  # noqa: ARG002, ANN002
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(transform=transform, dataset=dataset)

        csv_data = pd.read_csv(csv_path, parse_dates=["aerial_date"])
        target_cols = [
            col
            for col in csv_data.columns
            if col not in ("aerial_name", "aerial_date", "sen_name")
        ]

        aerial_dates = [
            aerial_date.to_pydatetime()
            for aerial_date in csv_data["aerial_date"].tolist()
        ]
        aerial_names = csv_data["aerial_name"].tolist()
        sen_names = csv_data["sen_name"].tolist()

        self.root_dir = root_dir

        self.aerial_names = aerial_names
        self.aerial_dates = aerial_dates
        self.sen_names = sen_names
        self.targets = csv_data[target_cols].to_numpy()

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        inputs = {}
        inputs["multilabel_classif"] = (self.targets[idx] > 0).astype(int)
        inputs["multilabel_classif_dates"] = self.aerial_dates[idx]

        meta = {}
        meta["s2_path"] = self.root_dir / "sentinel-ts" / self.sen_names[idx]
        meta["s2_h5_name"] = "sen-2-data"
        meta["s2_h5_mask"] = "sen-2-mask"
        meta["s1_asc_path"] = self.root_dir / "sentinel-ts" / self.sen_names[idx]
        meta["s1_asc_h5_name"] = "sen-1-asc-data"
        sen_path = self.root_dir / "sentinel-ts" / self.sen_names[idx]
        with h5py.File(sen_path, "r") as sen_file:
            for mod_h5, mod in self.MOD_MAPPING.items():
                meta[f"{mod}_dates"] = products_to_datetimes(
                    sen_file[f"{mod_h5}-products"][:],
                )

        meta["aerial_path"] = self.root_dir / "aerial" / self.aerial_names[idx]
        meta["aerial_dates"] = self.aerial_dates[idx]

        inputs_raster = super().preprocess_rasters(meta)
        inputs.update(inputs_raster)
        return self.transform(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.aerial_names)
