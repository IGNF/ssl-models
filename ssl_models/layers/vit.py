"""ViT building blocks, shared by the MAE and DINOv3 models."""

import math
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ssl_models.layers.rope import apply_rope


class Attention(nn.Module):
    """Multi-head self-attention with RoPE and optional key padding mask."""

    def __init__(
        self,
        dim: int,
        heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % heads:
            msg = f"Dim {dim} is not divisible by number of heads {heads}."
            raise ValueError(msg)
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(
        self,
        x: Tensor,
        rope: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply attention.

        Args:
            x: [B, L, C] tokens.
            rope: (sin, cos) of shape [L, 2 * head_dim] or [B, L, 2 * head_dim].
            attn_mask: [B, L] bool key padding mask, True = token is attended.
                Each sequence must keep at least one token (e.g. its CLS).

        """
        B, L, C = x.shape  # noqa: N806
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, heads, L, head_dim]
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, L, C)
        return self.proj(x)


class Mlp(nn.Module):
    """Two-layer GELU MLP."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward, hidden dim scaled by 2/3 to keep the param count."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = True,
        align_to: int = 8,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = align_to * math.ceil(hidden_dim / align_to)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def build_ffn(
    ffn: Literal["mlp", "swiglu"],
    dim: int,
    hidden_dim: int,
    bias: bool = True,
) -> nn.Module:
    """Build feed-forward layer."""
    match ffn:
        case "mlp":
            return Mlp(dim, hidden_dim, bias=bias)
        case "swiglu":
            return SwiGLUFFN(dim, hidden_dim, bias=bias)
        case _:
            msg = f"Invalid ffn {ffn}. Expected 'mlp' or 'swiglu'."
            raise ValueError(msg)


class LayerScale(nn.Module):
    """Learnable per-channel residual scaling."""

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class DropPath(nn.Module):
    """Stochastic depth: drop the residual branch per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class Block(nn.Module):
    """Pre-norm transformer block with LayerScale and stochastic depth."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ffn: Literal["mlp", "swiglu"] = "mlp",
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        layerscale_init: float | None = 1e-5,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = build_norm(norm, dim)
        self.attn = Attention(dim, heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.ls1 = (
            LayerScale(dim, layerscale_init) if layerscale_init else nn.Identity()
        )
        self.norm2 = build_norm(norm, dim)
        self.ffn = build_ffn(ffn, dim, int(dim * mlp_ratio), bias=ffn_bias)
        self.ls2 = (
            LayerScale(dim, layerscale_init) if layerscale_init else nn.Identity()
        )
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: Tensor,
        rope: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), rope, attn_mask)))
        return x + self.drop_path(self.ls2(self.ffn(self.norm2(x))))


class Transformer(nn.Module):
    """Stack of blocks, with an optional final norm (`final_norm`).

    `drop_path` is either a float (uniform rate) or a list with one rate per
    block, so that a backbone split into several stacks can apply a single
    linear drop path schedule across all of them.

    With `trunc_normal_init`, linear weights are initialized with a truncated
    normal of std 0.02 and zero biases; otherwise the PyTorch default init is
    kept (as in `vit_pytorch`).
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ffn: Literal["mlp", "swiglu"] = "mlp",
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        layerscale_init: float | None = 1e-5,
        drop_path: float | list[float] = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        final_norm: bool = False,
        trunc_normal_init: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(drop_path, float | int):
            drop_path = [float(drop_path)] * depth
        if len(drop_path) != depth:
            msg = f"Got {len(drop_path)} drop path rates for {depth} blocks."
            raise ValueError(msg)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    ffn=ffn,
                    norm=norm,
                    layerscale_init=layerscale_init,
                    drop_path=drop_path[idx],
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                )
                for idx in range(depth)
            ],
        )
        self.norm = build_norm(norm, dim) if final_norm else nn.Identity()
        if trunc_normal_init:
            self.apply(init_weights)

    def forward(
        self,
        x: Tensor,
        rope: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        for block in self.blocks:
            x = block(x, rope, attn_mask)
        return self.norm(x)


def build_norm(norm: Literal["layernorm", "rmsnorm"], dim: int) -> nn.Module:
    """Build normalization layer."""
    match norm:
        case "layernorm":
            return nn.LayerNorm(dim, eps=1e-5)
        case "rmsnorm":
            return nn.RMSNorm(dim, eps=1e-5)
        case _:
            msg = f"Invalid norm {norm}. Expected 'layernorm' or 'rmsnorm'."
            raise ValueError(msg)


def init_weights(module: nn.Module) -> None:
    """Init with truncated normal for linear layers, zero biases."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
