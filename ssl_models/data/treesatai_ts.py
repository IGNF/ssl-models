"""TreeSatAI-TS dataset."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from torch import Tensor
from torch.utils.data import Dataset

from ssl_models.data.utils import str_to_datetime


class TreeSatAITSDataset(Dataset):
    """TreeSatAI-TS dataset."""

    MOD_MAPPING: dict[str, str] = {  # noqa: RUF012
        "sen-2": "s2",
        "sen-1-asc": "s1_asc",
        "sen-1-des": "s1_des",
    }

    def __init__(
        self,
        root_dir: str | Path,
        csv_path: str | Path,
    ) -> None:
        csv_data = pd.read_csv(csv_path)
        cols = [col for col in csv_data.columns if col not in ("image", "aerial_dates")]

        aerial_dates = csv_data["dates"].tolist()
        aerial_names = csv_data["image"].tolist()
        aerial_stems = [Path(aerial_name).stem for aerial_name in aerial_names]
        h5_dir = Path(root_dir) / "sentinel-ts"
        h5_names = [
            next(h5_dir.glob(f"{aerial_stem}*.h5")) for aerial_stem in aerial_stems
        ]

        self.root_dir = Path(root_dir)
        self.aerial_dates = aerial_dates
        self.aerial_names = aerial_names
        self.h5_names = h5_names
        self.targets = csv_data[cols].to_numpy()

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get dataset item."""
        output = {}

        aerial_path = self.root_dir / "aerial" / self.aerial_names[idx]
        aerial_image = rasterio.open(aerial_path).read()
        output["aerial"] = aerial_image.astype(np.float32)

        aerial_dates = [self.aerial_dates[idx]]
        output["aerial_dates"] = str_to_datetime(aerial_dates)

        h5_path = self.root_dir / "sentinel-ts" / self.h5_names[idx]
        with h5py.File(h5_path, "r") as h5_file:
            sen_arrs = dict(h5_file.attrs.items())

        for mod_h5, mod in self.MOD_MAPPING.items():
            output[f"{mod}_dates"] = str_to_datetime(sen_arrs[f"{mod_h5}-products"])
            output[mod] = sen_arrs[f"{mod_h5}-data"]
            if mod_h5 == "sen-2":
                output[f"{mod}_masks"] = sen_arrs[f"{mod_h5}-masks"]

        output["multilabel_classif"] = self.targets[idx]
        return output

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.aerial_paths)
