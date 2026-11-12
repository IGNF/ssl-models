import pytest
import torch

from ssl_models.train.metric import MonoLabelMetric

# Rows are targets, columns are predictions.
CM = torch.tensor(
    [
        [8, 2, 0],
        [1, 3, 0],
        [0, 0, 0],  # class absent from the ground truth, never predicted
    ],
)


def cm_to_inputs(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build logits and targets reproducing a confusion matrix."""
    num_classes = cm.shape[0]
    preds, targets = [], []
    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            count = int(cm[true_cls, pred_cls])
            preds += [pred_cls] * count
            targets += [true_cls] * count
    logits = torch.nn.functional.one_hot(torch.tensor(preds), num_classes).float()
    return logits, torch.tensor(targets)


@pytest.fixture()
def metrics():
    metric = MonoLabelMetric(type_target="segment", num_classes=3)
    metric.update(*cm_to_inputs(CM))
    return metric.compute()


def test_confusion_matrix(metrics):
    assert torch.equal(metrics["confusion_matrix"], CM)


def test_per_class_metrics(metrics):
    # class 0: TP=8, FP=1, FN=2 / class 1: TP=3, FP=2, FN=1
    expected = {
        "per_class_iou": [8 / 11, 3 / 6],
        "per_class_f1": [16 / 19, 6 / 9],
        "per_class_precision": [8 / 9, 3 / 5],
        "per_class_recall": [8 / 10, 3 / 4],
    }
    for name, values in expected.items():
        torch.testing.assert_close(metrics[name][:2], torch.tensor(values))


def test_absent_class_is_nan(metrics):
    for name in (
        "per_class_iou",
        "per_class_f1",
        "per_class_precision",
        "per_class_recall",
    ):
        assert torch.isnan(metrics[name][2])
        assert not torch.isnan(metrics[name][:2]).any()


def test_averages_ignore_absent_class(metrics):
    for name in ("iou", "f1", "precision", "recall"):
        torch.testing.assert_close(
            metrics[f"average_{name}"],
            metrics[f"per_class_{name}"][:2].mean(),
        )
    torch.testing.assert_close(metrics["overall_accuracy"], torch.tensor(11 / 14))


def test_class_never_predicted_has_zero_precision():
    cm = torch.tensor([[5, 0], [3, 0]])
    metric = MonoLabelMetric(type_target="segment", num_classes=2)
    metric.update(*cm_to_inputs(cm))
    metrics = metric.compute()
    assert metrics["per_class_precision"][1] == 0
    assert metrics["per_class_recall"][1] == 0


def test_classif_keeps_legacy_metrics():
    metric = MonoLabelMetric(type_target="classif", num_classes=3)
    metric.update(*cm_to_inputs(CM))
    assert set(metric.compute()) == {"overall_accuracy", "confusion_matrix"}


def test_change_detect_has_two_classes():
    metric = MonoLabelMetric(type_target="change_detect", num_classes=1)
    logits = torch.tensor([5.0, 5.0, -5.0, -5.0, 5.0])
    targets = torch.tensor([1, 0, 0, 0, 1])
    metric.update(logits, targets)
    metrics = metric.compute()
    assert metrics["per_class_recall"].shape == (2,)
    torch.testing.assert_close(metrics["per_class_precision"][1], torch.tensor(2 / 3))
    torch.testing.assert_close(metrics["per_class_recall"][1], torch.tensor(1.0))
