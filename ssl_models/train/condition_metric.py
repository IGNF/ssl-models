"""ConditionMetric module."""

import torch
from torch import Tensor
from torchmetrics import Metric


class ConditionMetric(Metric):
    """Store condition metric."""

    def __init__(
        self,
        dim,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Init."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "mu",
            default=torch.zeros(dim),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "gram",
            default=torch.zeros(dim, dim),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(self, mu: Tensor, gram: Tensor, count: int) -> None:
        """Update mean and Gram matrix."""
        self.mu += mu * count
        self.gram += gram * count
        self.count += count

    def compute(self) -> tuple[Tensor, Tensor]:
        """Return aggregated weights."""
        return self.mu / self.count, self.gram / self.count
