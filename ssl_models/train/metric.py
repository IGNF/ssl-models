"""Code adapted from IGN's bdf-model."""

from typing import Literal

import torch
from torch import Tensor
from torchmetrics import AveragePrecision, Metric
from torchmetrics.classification import MultilabelF1Score
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
        threshold_detect: float = 0.5,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Init."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold_detect = threshold_detect

        self.f1_macro = MultilabelF1Score(
            num_labels=num_classes,
            threshold=self.threshold_detect,
            average="macro",
        )
        self.f1_weighted = MultilabelF1Score(
            num_labels=num_classes,
            threshold=self.threshold_detect,
            average="weighted",
        )
        self.ap_macro = AveragePrecision(
            task="multilabel",
            num_labels=num_classes,
            average="macro",
        )
        self.ap_weighted = AveragePrecision(
            task="multilabel",
            num_labels=num_classes,
            average="weighted",
        )

    def update(self, logits: Tensor, targets: Tensor) -> None:
        """Update metrics."""
        self.f1_macro.update(logits, targets.long())
        self.f1_weighted.update(logits, targets.long())
        self.ap_macro.update(logits, targets.long())
        self.ap_weighted.update(logits, targets.long())

    def compute(self) -> dict[str, Tensor]:
        """Compute metrics."""
        return {
            "macro_f1": self.f1_macro.compute(),
            "weighted_f1": self.f1_weighted.compute(),
            "macro_ap": self.ap_macro.compute(),
            "weighted_ap": self.ap_weighted.compute(),
        }

    def reset(self) -> None:
        self.f1_macro.reset()
        self.f1_weighted.reset()
        self.ap_macro.reset()
        self.ap_weighted.reset()
