"""Head layers."""

from abc import ABC
from functools import partial
from math import floor, log2
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from ssl_models.layers.embed import PixelifyBands


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
    ) -> Tensor:
        out = self.norm(x)

        q = rearrange(self.query, "(h d) -> 1 h 1 d", h=self.heads)
        k, v = self.to_kv(out).chunk(2, dim=-1)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h 1 d -> b (h d)")
        return self.norm_fc(out)


class ModalityWeightedReduce(nn.Module):
    """Reduce date/modality axis with learned per-modality weights."""

    def __init__(
        self,
        modality_sizes: list[int],
        modality_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.modality_sizes = modality_sizes
        self.modality_names = modality_names or [
            str(i) for i in range(len(modality_sizes))
        ]
        # Initialise to zeros -> softmax gives uniform weights at start
        self.modality_weights = nn.Parameter(torch.zeros(len(modality_sizes)))
        # Exposed after each forward for visualisation / logging
        self.last_modality_weights: Tensor | None = None

    def get_modality_weights(self) -> dict[str, float] | None:
        """Return last softmax-normalised weights as a name->float dict."""
        if self.last_modality_weights is None:
            return None
        return {
            name: self.last_modality_weights[i].item()
            for i, name in enumerate(self.modality_names)
        }

    def forward(self, x: Tensor) -> Tensor:
        """Reduce (B, D, L, C) to (B, L, C)."""
        chunks = x.split(self.modality_sizes, dim=1)  # list (B, D_m, L, C)
        mod_means = torch.stack(
            [c.mean(dim=1) for c in chunks],
            dim=0,
        )  # (num_mods, B, L, C)
        weights = torch.softmax(self.modality_weights, dim=0)  # (num_mods,)
        self.last_modality_weights = weights.detach()
        return (weights[:, None, None, None] * mod_means).sum(dim=0)  # (B, L, C)


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
            case "mean":
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
            case "mean":
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


class SegmentationHead(Head):
    """Convolutional decoder segmentation head.

    Reduces the date dimension, reshapes the token grid to a spatial feature
    map, then progressively upsamples with bilinear x2 stages (Upsample +
    Conv 3x3) to suppress patch-boundary artifacts.  Unlike ConvTranspose2d,
    the bilinear + conv approach avoids checkerboard artifacts and supports
    any integer upsample factor (not just powers of 2): the x2 stages cover
    the largest power-of-2 prefix; a final bilinear interpolation aligns the
    output to the exact ``target_size``.

    Parameters
    ----------
    type_head : {"mean", "attentive", "modweight"}
        How to reduce the date dimension.
        ``modweight`` learns one scalar weight per modality (softmax-normalised)
        and computes a weighted average of per-modality means.  The weights are
        exposed via :attr:`last_modality_weights` after each forward pass for
        visualisation.
    embed_dim : int
        Encoder embedding dimension.
    num_classes : int
        Number of output segmentation classes.
    grid_size : int
        Spatial size of the token grid (e.g. 16 for a 16x16 patch grid).
    target_size : int
        Output spatial resolution in pixels (e.g. 256).
    hidden_dim : int, default=256
        Number of channels in the first decoder stage.
    heads : int, default=8
        Number of attention heads (used when type_head="attentive").
    modality_sizes : list[int] | None
        Number of date tokens per modality in concatenation order.
        Required when type_head="modweight".
    modality_names : list[str] | None
        Name of each modality in the same order as modality_sizes.
        Used only for labelling :attr:`last_modality_weights`.
    """

    def __init__(
        self,
        type_head: Literal["mean", "attentive", "modweight"],
        embed_dim: int,
        num_classes: int,
        grid_size: int,
        target_size: int,
        hidden_dim: int = 256,
        heads: int = 8,
        modality_sizes: list[int] | None = None,
        modality_names: list[str] | None = None,
    ) -> None:
        super().__init__()

        upsample_factor = target_size / grid_size
        # Number of ×2 stages = floor(log2(upsample_factor)).
        # A final bilinear interpolation handles any non-power-of-2 remainder.
        num_ups = max(1, floor(log2(upsample_factor)))
        self.target_size = target_size
        self.type_head = type_head

        match type_head:
            case "mean":
                self.reduce = partial(torch.mean, dim=1)
            case "attentive":
                self.reduce = AttentiveReduce(dim=embed_dim, heads=heads)
            case "modweight":
                if modality_sizes is None:
                    msg = "modality_sizes is required when type_head='modweight'"
                    raise ValueError(msg)
                self.reduce = ModalityWeightedReduce(
                    modality_sizes=modality_sizes,
                    modality_names=modality_names,
                )

        self.proj = nn.Linear(embed_dim, hidden_dim)

        stages: list[nn.Module] = []
        in_ch = hidden_dim
        for i in range(num_ups):
            out_ch = max(hidden_dim >> (i + 1), num_classes * 2)
            stages += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GELU(),
                nn.GroupNorm(1, out_ch),
            ]
            in_ch = out_ch

        stages.append(nn.Conv2d(in_ch, num_classes, kernel_size=1))
        self.decoder = nn.Sequential(*stages)

    def get_modality_weights(self) -> dict[str, float] | None:
        """Return last softmax-normalised modality weights as a name→float dict.

        Returns ``None`` when type_head != "modweight" or before first forward.
        """
        if self.type_head != "modweight":
            return None
        return self.reduce.get_modality_weights()

    def forward(
        self,
        x: Tensor,
        ssl_phase: Literal["probe", "finetune"],
    ) -> Tensor:
        out = self.maybe_detach_features(x, ssl_phase)  # (B, D, L, C)
        b_size, _, num_tokens, _ = out.shape
        grid = round(num_tokens**0.5)

        # Reduce date dimension: (B, D, L, C) → (B, L, C)
        if self.type_head == "modweight":
            out = self.reduce(out)
        else:
            out = rearrange(out, "b d l c -> (b l) d c")
            out = self.reduce(out)  # (B*L, C)
            out = rearrange(out, "(b l) c -> b l c", b=b_size)

        # Project and reshape to spatial grid
        out = self.proj(out)  # (B, L, hidden_dim)
        out = rearrange(out, "b (h w) c -> b c h w", h=grid)  # (B, hidden_dim, G, G)

        # Convolutional upsampling
        out = self.decoder(out)

        # Final exact-size alignment (no-op when already at target_size)
        if out.shape[-1] != self.target_size:
            out = torch.nn.functional.interpolate(
                out,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )

        return rearrange(out, "b c h w -> b 1 c h w")
