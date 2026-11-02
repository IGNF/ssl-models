"""PASTIS-HD dataset module."""

from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.dataset.dataset import GenericDataset
from ssl_models.dataset.utils import dict_datetimes, read_csv, strs_datetimes


class PASTISHDDataset(GenericDataset):
    """PASTIS-HD dataset."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: Path,
        stage: Literal["train", "val", "test"],
        use_transform: bool,
        random_dates: bool,
        random_crop: bool,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__(
            dataset=dataset,
            stage=stage,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )
        csv_dir = Path(dataset.csv_dir) if dataset.csv_dir else root_dir

        csv_data = read_csv(
            csv_dir=csv_dir,
            stage=stage,
            ssl_phase=ssl_phase,
            balance_pretrain=dataset.balance_pretrain,
            val_pretrain=dataset.val_pretrain,
            filter_percent=dataset.filter_percent,
            fold=dataset.fold,
        )
        image_ids = csv_data["image"].tolist()

        meta_path = root_dir / "metadata.geojson"
        meta_data = gpd.read_file(meta_path).set_index("id")

        self.root_dir = root_dir
        self.image_ids = image_ids
        self.base_length = len(self.image_ids)
        self.repeats = dataset.repeats

        self.s2_dates = [
            dict_datetimes(meta_data.loc[str(image_id), "dates-S2"])
            for image_id in image_ids
        ]

        self.s1_asc_dates = [
            dict_datetimes(meta_data.loc[str(image_id), "dates-S1A"])
            for image_id in image_ids
        ]
        self.s1_des_dates = [
            dict_datetimes(meta_data.loc[str(image_id), "dates-S1D"])
            for image_id in image_ids
        ]

    def build_meta(self, idx: int) -> dict:
        """Return paths and dates of each modality."""
        image_id = self.image_ids[idx]
        spot_path = (
            self.root_dir
            / "DATA_SPOT"
            / "PASTIS_SPOT6_RVB_1M00_2019"
            / f"SPOT6_RVB_1M00_2019_{image_id}.tif"
        )
        spot_date = strs_datetimes(["2019-07-01"])

        meta = {}
        meta["spot_path"] = spot_path
        meta["spot_dates"] = spot_date

        meta["s2_path"] = self.root_dir / "DATA_S2" / f"S2_{image_id}.npy"
        meta["s2_dates"] = self.s2_dates[idx]
        meta["s1_asc_path"] = self.root_dir / "DATA_S1A" / f"S1A_{image_id}.npy"
        meta["s1_asc_dates"] = self.s1_asc_dates[idx]
        meta["s1_des_path"] = self.root_dir / "DATA_S1D" / f"S1D_{image_id}.npy"
        meta["s1_des_dates"] = self.s1_des_dates[idx]

        meta["pastis_seg_path"] = (
            self.root_dir / "ANNOTATIONS" / f"TARGET_{image_id}.npy"
        )
        meta["pastis_seg_dates"] = spot_date

        return meta

    def postprocess(
        self,
        inputs: dict[str, np.ndarray],
        meta: dict,
        idx: int,  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Set reference date."""
        inputs["ref_date"] = meta["spot_dates"]

        return inputs
