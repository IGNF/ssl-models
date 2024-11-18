"""Head layers."""

from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from ssl_models.layers.embed import Pixelify


class Head(nn.Module):
    """Abstract head class."""

    def maybe_detach_features(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        """Detach encoder features in probing phase."""
        if ssl_phase == "probe":
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

    def forward(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)
        out = self.norm(out)

        q = rearrange(self.query, "(h d) -> 1 h 1 d", h=self.heads)
        k, v = self.to_kv(out).chunk(2, dim=-1)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h 1 d -> b (h d)")
        return self.linear(out)


class LinearHead(Head):
    """Attentive head, with a mean pooling reduction."""

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)
        out = torch.mean(out, dim=1)
        return self.linear(out)


class PixelifyHead(Pixelify, Head):
    """Pixelify head."""

    def __init__(
        self,
        embed_dim: int,
        out_chans: int,
        patch_size: int,
    ) -> None:
        super().__init__(  # method from the inherited class listed first
            embed_dim=embed_dim,
            out_chans=out_chans,
            patch_size=patch_size,
            unpool=False,
        )

    def forward(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)
        out = out.unsqueeze(dim=1)  # unsqueeze date axis
        return super().forward(  # method from the inherited class listed first
            out,
            inds=[],
        )
