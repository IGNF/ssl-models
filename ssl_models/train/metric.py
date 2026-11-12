"""Metrics module."""

from typing import Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix
from torchmetrics.functional.classification.average_precision import (
    _multilabel_average_precision_compute,
)
from torchmetrics.functional.classification.precision_recall_curve import (
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_update,
)
from torchmetrics.utilities.data import dim_zero_cat

SEGMENT_METRIC_NAMES = [
    "overall_accuracy",
    "average_f1",
    "average_iou",
    "average_precision",
    "average_recall",
    "per_class_f1",
    "per_class_iou",
    "per_class_precision",
    "per_class_recall",
    "confusion_matrix",
]


def _safe_div(num: Tensor, den: Tensor) -> Tensor:
    """Divide, returning 0 where the denominator is 0."""
    return torch.where(den > 0, num / den.clamp(min=1), torch.zeros_like(num))


class MonoLabelMetric(Metric):
    """Store confusion matrix."""

    def __init__(
        self,
        type_target: Literal["classif", "segment", "change_detect"],
        num_classes: int | None,
        threshold_detect: float = 0.5,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Init."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold_detect = threshold_detect

        match type_target:
            case "classif":
                self.task = "multiclass"
                self.num_classes = num_classes
                self.metric_names = ["overall_accuracy", "confusion_matrix"]
            case "segment":
                self.task = "multiclass"
                self.num_classes = num_classes
                self.metric_names = SEGMENT_METRIC_NAMES
            case "change_detect":
                self.task = "binary"
                self.num_classes = 2
                self.metric_names = SEGMENT_METRIC_NAMES

        self.add_state(
            "cm",
            default=torch.zeros(self.num_classes, self.num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, logits: Tensor, targets: Tensor) -> None:
        """Update confusion matrix."""
        match self.task:
            case "classif" | "segment":
                logits = logits.argmax(dim=1)
            case "change_detect":
                logits = (torch.sigmoid(logits) > self.threshold_detect).long()

        self.cm += confusion_matrix(
            logits,
            targets,
            task=self.task,
            num_classes=self.num_classes,
            normalize=None,
        )

    def compute(self) -> dict[str, Tensor]:
        """Compute metrics."""
        overall_acc = self.cm.trace() / self.cm.sum()  # OA
        true_pos = torch.diag(self.cm).float()  # TP
        false_pos = self.cm.sum(0).float() - true_pos  # FP
        false_neg = self.cm.sum(1).float() - true_pos  # FN

        per_class = {
            "f1": _safe_div(2 * true_pos, 2 * true_pos + false_pos + false_neg),
            "iou": _safe_div(true_pos, true_pos + false_pos + false_neg),
            "precision": _safe_div(true_pos, true_pos + false_pos),
            "recall": _safe_div(true_pos, true_pos + false_neg),
        }

        # Classes absent from the ground truth are excluded from averages.
        valid_classes = (true_pos + false_neg) > 0
        metrics = {"overall_accuracy": overall_acc, "confusion_matrix": self.cm}
        for name, values in per_class.items():
            metrics[f"average_{name}"] = values[valid_classes].mean()
            metrics[f"per_class_{name}"] = torch.where(
                valid_classes,
                values,
                torch.full_like(values, torch.nan),
            )
        return {
            metric_name: metric_val
            for metric_name, metric_val in metrics.items()
            if metric_name in self.metric_names
        }


class MultiLabelMetric(Metric):
    """Store confusion matrix and precision recall curves."""

    def __init__(
        self,
        num_labels: int,
        threshold_detect: float = 0.5,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_labels = num_labels
        self.threshold_detect = threshold_detect

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state(
            "cm",
            default=torch.zeros(num_labels, 2, 2, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, logits: Tensor, targets: Tensor) -> None:
        """Update metrics."""
        logits = torch.sigmoid(logits)
        targets = targets.long()

        preds, target, _ = _multilabel_precision_recall_curve_format(
            logits,
            targets,
            num_labels=self.num_labels,
            thresholds=None,
            ignore_index=None,
        )
        state = _multilabel_precision_recall_curve_update(
            preds,
            target,
            num_labels=self.num_labels,
            thresholds=None,
        )

        self.preds.append(state[0])
        self.target.append(state[1])

        self.cm += confusion_matrix(
            logits > self.threshold_detect,
            targets,
            task="multilabel",
            num_labels=self.num_labels,
            normalize=None,
        )

    def compute(self) -> Tensor:
        """Compute metrics."""
        true_pos = self.cm[:, 1, 1]  # TP
        false_pos = self.cm[:, 0, 1]  # FP
        false_neg = self.cm[:, 1, 0]  # FN
        label_weights = (true_pos + false_neg) / (true_pos + false_neg).float().sum()

        per_label_f1 = (2 * true_pos) / (2 * true_pos + false_pos + false_neg)
        per_label_ap = _multilabel_average_precision_compute(
            (dim_zero_cat(self.preds), dim_zero_cat(self.target)),
            self.num_labels,
            average="none",
            thresholds=None,
            ignore_index=None,
        )
        return {
            "average_f1": per_label_f1.nanmean(),  # macro F1
            "average_ap": per_label_ap.nanmean(),  # macro mAP
            "weighted_f1": (per_label_f1 * label_weights).nansum(),
            "weighted_ap": (per_label_ap * label_weights).nansum(),
        }
