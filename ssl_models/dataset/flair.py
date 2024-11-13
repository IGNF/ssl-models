"""FLAIR dataset module."""


from pathlib import Path

import albumentations as A  # noqa: N812
import numpy as np
import pandas as pd

from conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset


class FLAIRDataset(GenericDataset):
    """FLAIR dataset."""

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

        csv_data = pd.read_csv(csv_path)

        patch_domains = csv_data["domains"]
        patch_areas = csv_data["areas"]

        self.root_dir = root_dir
        self.patch_domains = patch_domains
        self.patch_areas = patch_areas

        def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
            """Get dataset item."""
            patch_name = self.patch_domains[idx] + "_" + self.patch_areas[idx]

            meta = {}
            modalities_tif = ["aerial", "dsm", "dtm", "label", "spot"]
            for mod in modalities_tif:
                meta[f"{mod}_path"] = self.root_dir / f"{self.patch_domains[idx]}-{mod.upper()}" / self.patch_areas[idx] / f"{patch_name}_{mod.upper()}-DATA.tif"


        def __len__(self) -> int:
            """Return length of dataset."""
            return len(self.patch_domains)