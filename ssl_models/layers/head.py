"""Head layers."""

from abc import ABC
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn

from ssl_models.layers.embed import PixelifyBands
from ssl_models.layers.vit import init_weights


class Head(nn.Module, ABC):
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


class AttentiveReduce(nn.Module):
    """Attentive reduction with self-attention."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.norm_fc = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)

        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.query = nn.Parameter(torch.randn(dim))

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Reduce x [B, L, C] to [B, C], over the tokens with mask [B, L] True
        (all of them if None)."""
        out = self.norm(x)

        q = rearrange(self.query, "(h d) -> 1 h 1 d", h=self.heads)
        k, v = self.to_kv(out).chunk(2, dim=-1)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = dots.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h 1 d -> b (h d)")
        return self.norm_fc(out)


class ClassificationHead(Head):
    """Classification head."""

    def __init__(
        self,
        type_head: Literal["mean", "attentive"],
        dim: int,
        num_classes: int,
        heads: int = 8,
    ) -> None:
        super().__init__()

        match type_head:
            case "linear":
                self.reduce = partial(torch.mean, dim=1)
            case "attentive":
                self.reduce = AttentiveReduce(dim=dim, heads=heads)

        self.linear = nn.Linear(dim, num_classes)

    def forward(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)
        out = self.reduce(out)
        return self.linear(out)


class PixelifyHead(PixelifyBands, Head):
    """Pixelify head."""

    def __init__(
        self,
        type_head: Literal["mean", "attentive"],
        dim: int,
        out_chans: int,
        patch_size: int,
        heads: int = 8,
    ) -> None:
        super().__init__(  # method used is from the first listed inherited class
            embed_dim=dim,
            out_chans=out_chans,
            patch_size=patch_size,
        )

        match type_head:
            case "linear":
                self.reduce = partial(torch.mean, dim=1)
            case "attentive":
                self.reduce = AttentiveReduce(dim=dim, heads=heads)

    def forward(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)
        out = rearrange(out, "b d l c -> (b l) d c")
        out = self.reduce(out)
        out = rearrange(out, "(b l) c -> b 1 l c", l=x.shape[2])
        return super().forward(  # method used is from the first listed inherited class
            out,
        )


class DINOHead(nn.Module):
    """MLP projection, L2-normalized bottleneck, then cosine to prototypes.

    The prototypes are L2-normalized in the forward pass, which is equivalent
    to a weight-normalized last layer with fixed unit norm. Logits are thus
    cosine similarities in [-1, 1].
    """

    def __init__(
        self,
        in_dim: int,
        num_prototypes: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers == 1:
            layers = [nn.Linear(in_dim, bottleneck_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(init_weights)

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, bottleneck_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = F.normalize(self.mlp(x), dim=-1, eps=1e-6)
        return F.linear(x, F.normalize(self.prototypes, dim=-1, eps=1e-6))
