"""Code adapted from IGN's bdf-model."""

from typing import Literal

import torch
from torch import Tensor
from torchmetrics import AveragePrecision, Metric
from torchmetrics.functional import confusion_matrix

SMOOTH = 0.000001


class MonoLabelMetric(Metric):
    """Store metrics as a confusion matrix."""

    def __init__(
        self,
        type_target: Literal["classif", "segment", "change_detect"],
        num_classes: int | None,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Init."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        match type_target:
            case "classif":
                self.task = "multiclass"
                self.num_classes = num_classes
                self.metric_names = ["overall_accuracy"]
            case "segment":
                self.task = "multiclass"
                self.num_classes = num_classes
                self.metric_names = ["overall_accuracy", "average_f1", "average_iou"]
            case "change_detect":
                self.task = "binary"
                self.num_classes = 2
                self.metric_names = ["overall_accuracy", "average_f1", "average_iou"]

        self.add_state(
            "cm",
            default=torch.zeros(self.num_classes, self.num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, logits: Tensor, targets: Tensor) -> None:
        """Update confusion matrix."""
        self.cm += confusion_matrix(
            logits.argmax(dim=1),
            targets,
            task=self.task,
            num_classes=self.num_classes,
            normalize=None,
        )

    def compute(self) -> dict[str, Tensor]:
        """Compute metrics from confusion matrix."""
        overall_acc = self.cm.trace() / self.cm.sum()  # OA
        true_pos = torch.diag(self.cm)  # TP
        false_pos = self.cm.sum(0) - torch.diag(self.cm)  # FP
        false_neg = self.cm.sum(1) - torch.diag(self.cm)  # FN

        per_class_f1 = (2 * true_pos) / (2 * true_pos + false_pos + false_neg + SMOOTH)
        per_class_iou = true_pos / (true_pos + false_pos + false_neg + SMOOTH)
        metrics = {
            "overall_accuracy": overall_acc,
            "average_f1": per_class_f1.mean(),
            "average_iou": per_class_iou.mean(),
        }
        return {
            metric_name: metric_val
            for metric_name, metric_val in metrics.items()
            if metric_name in self.metric_names
        }


class MultiLabelMetric(Metric):
    """Store MAP metric and confusion matrix."""

    def __init__(
        self,
        num_classes: int,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Init."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "cm",
            default=torch.zeros(num_classes, 2, 2, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.average_precision = AveragePrecision(
            task="multilabel",
            num_labels=num_classes,
        )

    def update(self, logits: Tensor, targets: Tensor) -> None:
        """Update confusion matrix."""
        self.cm += confusion_matrix(
            logits > 0,
            targets,
            task="multilabel",
            num_labels=targets.shape[1],
            normalize=None,
        )
        self.average_precision.update(torch.sigmoid(logits), targets)

    def compute(self) -> dict[str, Tensor]:
        """Compute metrics from confusion matrix."""
        true_pos = self.cm[:, 1, 1]  # TP
        false_pos = self.cm[:, 0, 1]  # FP
        false_neg = self.cm[:, 1, 0]  # FN

        per_class_f1 = (2 * true_pos) / (2 * true_pos + false_pos + false_neg + SMOOTH)
        per_class_iou = true_pos / (true_pos + false_pos + false_neg + SMOOTH)
        per_class_ap = self.average_precision.compute()

        return {
            "average_f1": per_class_f1.mean(),
            "average_iou": per_class_iou.mean(),
            "map": per_class_ap.mean(),
        }

    def reset(self) -> None:
        super().reset()
        self.average_precision.reset()
