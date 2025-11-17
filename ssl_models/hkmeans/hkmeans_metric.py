"""HKMeansMetric module."""

import torch
from torch import Tensor
from torchmetrics import Metric

from ssl_models.hkmeans import hierarchical_kmeans_gpu as hkmg
from ssl_models.hkmeans import hierarchical_sampling as hs
from ssl_models.hkmeans.clusters import HierarchicalCluster


class HKMeansMetric(Metric):
    """Store confusion matrix."""

    def __init__(
        self,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Init."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("ids", default=[], dist_reduce_fx="cat")
        self.add_state("cls_token", default=[], dist_reduce_fx="cat")
        self.add_state("weight", default=[], dist_reduce_fx="cat")

    def update(self, ids: Tensor, cls_token: Tensor) -> None:
        """Update cls_token and ids."""
        self.ids.append(ids)
        self.cls_token.append(cls_token)
        self.weight.append(torch.zeros_like(ids, dtype=torch.float32))

    def compute_local(self) -> None:
        """Compute HKMeans weights."""
        self.ids = [torch.cat(self.ids)]
        self.cls_token = [torch.cat(self.cls_token)]
        self.weight = [torch.cat(self.weight)]

        size = self.ids[0].shape[0]
        clusters = hkmg.hierarchical_kmeans_with_resampling(
            data=self.cls_token[0].to(dtype=torch.float32),
            n_clusters=[size // 10, size // 100, size // 1000],
            n_levels=3,
            sample_sizes=[3, 3, 3],
            n_resamples=10,
            verbose=True,
            sample_strategy="closest",
        )
        cl = HierarchicalCluster.from_dict(clusters)
        for _ in range(100):
            indices = hs.hierarchical_sampling(cl, target_size=size // 10)
            self.weight[0][indices] += 1
        self.weight[0] = torch.maximum(self.weight[0], self.weight[0].max() / 10)
        self.weight[0] /= self.weight[0].mean()

    def compute(self) -> tuple[Tensor, Tensor]:
        """Return aggregated weights."""
        return self.ids, self.weight
