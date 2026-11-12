from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from ssl_models.train.logger import MetricsLogger

VALUES = torch.tensor([0.5, torch.nan, 0.25])


def log_per_class(class_names=None, stage="val", **trainer_options):
    trainer = SimpleNamespace(
        ssl_phase="finetune",
        log_per_class_metrics=True,
        per_class_metrics=["iou", "f1"],
        per_class_stages=["val", "test"],
    )
    for name, value in trainer_options.items():
        setattr(trainer, name, value)
    pl_module = MagicMock()
    MetricsLogger().log_per_class_metric(
        trainer=trainer,
        pl_module=pl_module,
        name_target="cosia",
        name_metric="iou",
        values=VALUES,
        class_names=class_names,
        stage=stage,
    )
    return [call.kwargs["name"] for call in pl_module.log.call_args_list]


def test_logs_named_classes_and_skips_absent():
    names = log_per_class(class_names=["building", "swimming pool", "water/sea"])
    assert names == [
        "finetune_cosia/iou_building_val",
        "finetune_cosia/iou_water_sea_val",
    ]


def test_falls_back_to_indices():
    assert log_per_class(class_names=None) == [
        "finetune_cosia/iou_0_val",
        "finetune_cosia/iou_2_val",
    ]
    assert log_per_class(class_names=["a"]) == [
        "finetune_cosia/iou_0_val",
        "finetune_cosia/iou_2_val",
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"log_per_class_metrics": False},
        {"per_class_stages": ["test"]},
        {"per_class_metrics": ["f1"]},
    ],
)
def test_disabled_options_log_nothing(options):
    assert log_per_class(**options) == []
