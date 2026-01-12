"""ConditionMetric module."""

import torch
from torch import Tensor
from torchmetrics import Metric


class ConditionMetric(Metric):
    """Store condition metric."""

    def __init__(
        self,
        dim: int,
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


class BalanceMetric(Metric):
    """Compute balancing based on rank maximization."""

    def __init__(
        self,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Init."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.mu, self.gram = {}, {}

        self.add_state("ids", default=[], dist_reduce_fx="cat")
        self.add_state("weight", default=[], dist_reduce_fx="cat")

    def update(
        self, ids: Tensor, mu: dict[str, Tensor], gram: dict[str, Tensor],
    ) -> None:
        """Update."""
        for name_mod in mu:
            if name_mod not in self.mu:
                self.mu[name_mod] = []
                self.gram[name_mod] = []
            self.mu[name_mod].append(mu[name_mod])
            self.gram[name_mod].append(gram[name_mod])
        self.ids.append(ids)
        self.weight.append(torch.ones_like(ids, dtype=torch.float32))

    def compute_local(self, max_balance: float) -> None:
        """Compute balancing weights."""
        self.ids = [torch.cat(self.ids)]
        self.weight = [torch.cat(self.weight)]

        self.mu = {name_mod: torch.cat(self.mu[name_mod]) for name_mod in self.mu}
        self.gram = {name_mod: torch.cat(self.gram[name_mod]) for name_mod in self.gram}

        # create a temporary parameter starting from current state
        p = torch.nn.Parameter(self.weight[0].clone())

        # local optimizer for this compute call
        optimizer = torch.optim.Adam([p])
        prev_loss = None
        tol = 1e-4

        for idx in range(100000):
            loss = 0
            for name_mod in self.mu:
                mu = (p[:, None] * self.mu[name_mod]).mean(dim=0) / p.mean()
                gram = (p[:, None, None] * self.gram[name_mod]).mean(dim=0) / p.mean()
                cov = gram - mu[:, None] * mu[None, :]
                _, S, _ = torch.linalg.svd(cov, full_matrices=False)
                loss -= (torch.sum(S) ** 2) / torch.sum(S**2)

            # backward and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                p.clamp_(min=1/max_balance, max=max_balance)
                p.div_(p.mean())

            if idx % 50 == 0:
                print(loss)
                if prev_loss is not None and prev_loss - loss < tol:
                    break
                prev_loss = loss


        self.weight[0].copy_(p.detach())

    def compute(self) -> tuple[Tensor, Tensor]:
        """Return aggregated weights."""
        return self.ids, self.weight
