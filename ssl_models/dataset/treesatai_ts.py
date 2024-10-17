"""TreeSatAI-TS dataset."""

from pathlib import Path

import albumentations as A  # noqa: N812
import h5py
import numpy as np
import pandas as pd
import rasterio

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import str_to_datetime


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
    ) -> None:
        super().__init__(transform=transform, dataset=dataset)

        csv_data = pd.read_csv(csv_path)

        aerial_dates = csv_data["dates"].tolist()
        aerial_names = csv_data["image"].tolist()
        h5_dir = Path(root_dir) / "sentinel-ts"
        h5_names = [
            next(h5_dir.glob(f"{aerial_name.stem}*.h5")).name
            for aerial_name in aerial_names
        ]
        target_cols = [
            col for col in csv_data.columns if col not in ("image", "aerial_dates")
        ]

        self.root_dir = root_dir

        self.aerial_dates = aerial_dates
        self.aerial_names = aerial_names
        self.h5_names = h5_names
        self.targets = csv_data[target_cols].to_numpy()

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get dataset item."""
        inputs = {}

        aerial_dates = [self.aerial_dates[idx]]
        inputs["aerial_dates"] = str_to_datetime(aerial_dates)

        aerial_path = self.root_dir / "aerial" / self.aerial_names[idx]
        aerial = rasterio.open(aerial_path).read()
        inputs["aerial"] = aerial[
            :,
            self.SHIFT_AERIAL[0] : -self.SHIFT_AERIAL[0],
            self.SHIFT_AERIAL[1] : -self.SHIFT_AERIAL[1],
        ]

        h5_path = self.root_dir / "sentinel-ts" / self.h5_names[idx]
        with h5py.File(h5_path, "r") as h5_file:
            sen_arrs = dict(h5_file.attrs.items())

        for mod_h5, mod in self.MOD_MAPPING.items():
            inputs[f"{mod}_dates"] = str_to_datetime(sen_arrs[f"{mod_h5}-products"])
            inputs[mod] = sen_arrs[f"{mod_h5}-data"]
            if mod_h5 == "sen-2":
                inputs[f"{mod}_mask"] = sen_arrs[f"{mod_h5}-masks"]

        inputs["regress"] = self.targets[idx]
        inputs["multilabel_classif"] = self.targets[idx] > 0
        return super().preprocess(inputs)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.aerial_names)
