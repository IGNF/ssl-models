from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torchmetrics import MeanMetric, Metric
from torchmetrics.classification import MulticlassAccuracy

from ssl_models.geopile_data import GeoPileDataset


class ClassifMetric(Metric):
    """Subclass of torchmetrics Metric.

    Stores aggregated accuracies for different datasets.
    """

    def __init__(self) -> None:
        super().__init__(dist_sync_on_step=False)

        self.num_classes = GeoPileDataset.num_classes
        self.missing_val = GeoPileDataset.missing_val
        self.cross_entropy = CrossEntropyLoss(ignore_index=self.missing_val)
        self.acc_metric = nn.ModuleDict(
            [
                [
                    dataset,
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=self.missing_val,
                    ),
                ]
                for dataset, num_classes in self.num_classes.items()
            ],
        )
        self.agg_loss = nn.ModuleDict(
            [[dataset, MeanMetric()] for dataset in self.num_classes],
        )
        self.agg_acc = nn.ModuleDict(
            [[dataset, MeanMetric()] for dataset in self.num_classes],
        )

    def update(self, head_logits: dict[str, Tensor], batch: dict[str, Tensor]) -> None:
        """Update aggregate losses and accuracies."""
        for dataset in self.num_classes:
            if torch.all(batch[f"target_{dataset}"] == self.missing_val):
                continue
            weight = (batch[f"target_{dataset}"] != self.missing_val).float().mean()

            loss = self.cross_entropy(
                head_logits[dataset],
                batch[f"target_{dataset}"],
            )
            self.agg_loss[dataset].update(loss, weight)

            acc = self.acc_metric[dataset](
                head_logits[dataset],
                batch[f"target_{dataset}"],
            )
            self.agg_acc[dataset].update(acc, weight)

    def compute_loss(
        self,
        head_logits: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> Tensor:
        """Compute loss."""
        sum_loss = 0
        for dataset in self.num_classes:
            if torch.all(batch[f"target_{dataset}"] == self.missing_val):
                continue
            loss = self.cross_entropy(
                head_logits[dataset],
                batch[f"target_{dataset}"],
            )
            sum_loss += loss
        return sum_loss

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute aggregate loss and accuracies."""
        agg_loss = 0
        for dataset in self.num_classes:
            loss = self.agg_loss[dataset].compute()
            agg_loss += loss

        agg_acc = {
            dataset: self.agg_acc[dataset].compute() for dataset in self.num_classes
        }
        return agg_loss, agg_acc
