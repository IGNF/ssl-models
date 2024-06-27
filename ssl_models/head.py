"""Head layers."""

import torch
from einops import rearrange
from torch import Tensor, nn


class Head(nn.Module):
    """Abstract head class."""

    def maybe_detach_features(self, x: Tensor, ssl_phase: str) -> Tensor:
        """Detach encoder features in probing phase."""
        if ssl_phase == "probe":
            x = x.detach()
        return x

    def maybe_detach_prediction(self, x: Tensor, ssl_phase: str) -> Tensor:
        """Detach head prediction in pretraining phase."""
        if ssl_phase == "pretrain":
            x = x.detach()
        return x


class AttentiveHead(Head):
    """Attentive head, with a self-attention reduction."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.query = nn.Parameter(torch.randn(inner_dim))

        self.linear = nn.Linear(inner_dim, num_classes)

    def forward(self, x: Tensor, ssl_phase: int) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)
        out = self.norm(out)

        q = rearrange(self.query, "(H D) -> 1 H 1 D", H=self.heads)
        k, v = self.to_kv(out).chunk(2, dim=-1)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.heads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "B H 1 D -> B (H D)")
        out = self.linear(out)
        return self.maybe_detach_prediction(out, ssl_phase)


class LinearHead(Head):
    """Attentive head, with a mean pooling reduction."""

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor, ssl_phase: str) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)
        out = torch.mean(out, dim=1)
        out = self.linear(out)
        return self.maybe_detach_prediction(out, ssl_phase)
