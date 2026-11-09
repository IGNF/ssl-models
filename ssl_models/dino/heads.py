"""DINO / iBOT projection head."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ssl_models.layers.vit import init_weights


class DINOHead(nn.Module):
    """MLP projection, L2-normalized bottleneck, then cosine to prototypes.

    The prototypes are L2-normalized in the forward pass, which is equivalent
    to a weight-normalized last layer with fixed unit norm. Logits are thus
    cosine similarities in [-1, 1].
    """

    def __init__(
        self,
        in_dim: int,
        n_prototypes: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        if n_layers == 1:
            layers = [nn.Linear(in_dim, bottleneck_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(init_weights)

        self.prototypes = nn.Parameter(torch.empty(n_prototypes, bottleneck_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = F.normalize(self.mlp(x), dim=-1, eps=1e-6)
        return F.linear(x, F.normalize(self.prototypes, dim=-1, eps=1e-6))
