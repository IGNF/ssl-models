import os
from pathlib import Path

import pytest
import torch

from ssl_models.baselines.dinov3 import Dinov3Baseline
from ssl_models.conf.dataset.cosia_histo import COSIAHistoConfig
from ssl_models.conf.datasets import DatasetsConfig


def _build_test_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    aerial = torch.rand(batch_size, 1, 4, 64, 64, dtype=torch.float32)
    date = torch.tensor([[[2020, 150, 0]]], dtype=torch.int16).repeat(batch_size, 1, 1)
    return {
        "aerial": aerial,
        "aerial_dates": date,
        "ref_date": date,
    }


@pytest.mark.skipif(
    not os.getenv("DINOV3_REPO_DIR"),
    reason="Set DINOV3_REPO_DIR to run Dinov3 forward integration test.",
)
def test_dinov3_forward_with_local_weights(tmp_path: Path) -> None:
    """Forward smoke test for Dinov3Baseline with local hub repo + checkpoint.

    Required environment variables:
    - DINOV3_REPO_DIR: local path containing hubconf.py

    Optional environment variables:
    - DINOV3_WEIGHTS_PATH: path to .pth weights file
    - DINOV3_HUB_MODEL: hub entrypoint name (default inferred from backbone_type)
    - DINOV3_BACKBONE_TYPE: web_base | web_large | sat_large (default web_base)
    """
    repo_dir = os.environ["DINOV3_REPO_DIR"]
    weights_path = os.getenv("DINOV3_WEIGHTS_PATH")
    hub_model = os.getenv("DINOV3_HUB_MODEL")
    backbone_type = os.getenv("DINOV3_BACKBONE_TYPE", "web_base")

    if weights_path is not None and not Path(weights_path).exists():
        msg = f"DINOV3_WEIGHTS_PATH does not exist: {weights_path}"
        raise FileNotFoundError(msg)

    cosia_histo = COSIAHistoConfig(
        rel_dir="",
        crop_meters=51.2,
    )
    cosia_histo.aerial.image_size = 256

    datasets = DatasetsConfig(
        root_dir=tmp_path,
        name_datasets=["cosia_histo"],
        cosia_histo=cosia_histo,
    )

    model = Dinov3Baseline(
        datasets=datasets,
        backbone_type=backbone_type,
        checkpoint_path=weights_path,
        repo_dir=repo_dir,
        hub_model=hub_model,
        fusion_mode="shared",
        type_head="linear",
        interpolate="nearest",
        add_date_enc=False,
        keep_norm=False,
        resize=False,
    )
    model.eval()

    batch = _build_test_batch(batch_size=1)
    with torch.no_grad():
        _, _, _, logits = model(batch, ssl_phase="finetune")

    assert logits is not None
    assert "cosia" in logits
    assert logits["cosia"].shape[0] == 1
    assert logits["cosia"].shape[1] == cosia_histo.cosia.num_classes