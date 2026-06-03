from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from ssl_models.conf.dataset.cosia_histo import COSIAHistoConfig
from ssl_models.dataset.cosia_histo import COSIAHistoDataset


def test_cosia_histo_dataset_train_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    root_dir = tmp_path / "root"
    root_dir.mkdir(parents=True, exist_ok=True)

    modalities = gpd.GeoDataFrame(
        {
            "fid": [1, 2, 3],
            "aerial_rgbi": ["aerial/patch_1.tif", "aerial/patch_2.tif", "aerial/patch_3.tif"],
            "ground_truth": ["labels/label_1.tif", "labels/label_2.tif", "labels/label_3.tif"],
        },
    )
    split = gpd.GeoDataFrame(
        {
            "fid": [1, 2, 3],
            "split": ["train", "val", "train"],
        },
    )

    def fake_read_file(path: Path, **kwargs):  # noqa: ARG001, ANN202
        if "MODALITIES" in str(path):
            return modalities
        if "SPLIT" in str(path):
            return split
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(gpd, "read_file", fake_read_file)

    def fake_preprocess_rasters(self, meta, start_gcd=None):  # noqa: ARG001, ANN202
        assert meta["aerial_path"] == root_dir / "aerial/patch_1.tif"
        assert meta["cosia_path"] == root_dir / "labels/label_1.tif"
        assert meta["aerial_dates"].shape == (1, 3)
        assert meta["cosia_dates"].shape == (1, 3)
        return {
            "aerial": np.zeros((1, 4, 512, 512), dtype=np.float32),
            "cosia": np.zeros((1, 1, 512, 512), dtype=np.float32),
            "aerial_dates": meta["aerial_dates"],
            "cosia_dates": meta["cosia_dates"],
        }

    monkeypatch.setattr(COSIAHistoDataset, "preprocess_rasters", fake_preprocess_rasters)

    dataset_cfg = COSIAHistoConfig(
        rel_dir="",
        repeats=2,
        modalities_gpkg="GLOBAL_ALL_MTD_MODALITIES",
        split_gpkg="GLOBAL_ALL_MTD_SPLIT",
        fid_col="fid",
        split_col="split",
        aerial_path_col="aerial_rgbi",
        target_path_col="ground_truth",
        default_date="20000101",
        default_date_fmt="%Y%m%d",
    )

    dataset = COSIAHistoDataset(
        dataset=dataset_cfg,
        root_dir=root_dir,
        stage="train",
        use_transform=False,
        random_dates=False,
        random_crop=False,
        ssl_phase="probe",
    )

    assert dataset.base_length == 2
    assert len(dataset) == 8

    sample = dataset[0]

    assert sample["aerial"].shape == (1, 4, 512, 512)
    assert sample["cosia"].shape == (1, 1, 512, 512)
    assert sample["ref_date"].shape == (1, 3)