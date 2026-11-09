"""ViT building blocks for DINOv3-style models.

Own implementation (no external DINOv3 code): 2D axial RoPE with coordinate
augmentations, attention with optional key padding mask, GELU MLP / SwiGLU
feed-forward, LayerNorm / RMSNorm, LayerScale and stochastic depth.

RoPE is computed from explicit patch coordinates normalized to [-1, 1] within
a view, so that tokens of different modalities covering the same location get
the same rotation, whatever their grid size. Prefix tokens (CLS, registers)
get the identity rotation (see `pad_rope`).
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

Rope = tuple[Tensor, Tensor]  # (sin, cos), each [..., N, head_dim]


def grid_coords(
    h: int,
    w: int,
    device: torch.device | None = None,
) -> Tensor:
    """Patch center coordinates of an h x w grid, normalized to [-1, 1].

    Returns a [h * w, 2] tensor of (y, x) coordinates in row-major order.
    """
    y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    yy, xx = torch.meshgrid(2 * y - 1, 2 * x - 1, indexing="ij")
    return torch.stack([yy.flatten(), xx.flatten()], dim=-1)


class RopePositionEmbedding(nn.Module):
    """2D axial rotary position embedding from explicit coordinates.

    Half of the head dimension rotates with the y coordinate, the other half
    with x. At train time, coordinates can be augmented (same draw for the
    whole forward call, hence consistent across modalities of a view):
        - shift: uniform translation in [-shift, shift] per axis,
        - jitter: log-uniform scaling in [1/jitter, jitter] per axis,
        - rescale: log-uniform isotropic scaling in [1/rescale, rescale].
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 100.0,
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 4:
            msg = f"RoPE head dim must be a multiple of 4, got {head_dim}."
            raise ValueError(msg)
        self.head_dim = head_dim
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        exponents = (
            2 * torch.arange(head_dim // 4, dtype=torch.float32) / (head_dim // 2)
        )
        self.register_buffer("periods", base**exponents, persistent=False)

    def augment(self, coords: Tensor) -> Tensor:
        """Apply train-time coordinate augmentations."""
        device = coords.device
        if self.shift_coords:
            shift = torch.empty(2, device=device).uniform_(
                -self.shift_coords,
                self.shift_coords,
            )
            coords = coords + shift
        if self.jitter_coords:
            log_jitter = math.log(self.jitter_coords)
            jitter = torch.empty(2, device=device).uniform_(-log_jitter, log_jitter)
            coords = coords * jitter.exp()
        if self.rescale_coords:
            log_rescale = math.log(self.rescale_coords)
            rescale = torch.empty(1, device=device).uniform_(
                -log_rescale,
                log_rescale,
            )
            coords = coords * rescale.exp()
        return coords

    def forward(self, coords: Tensor) -> Rope:
        """Compute (sin, cos) of shape [..., N, head_dim] from [..., N, 2] coords."""
        coords = coords.float()
        if self.training:
            coords = self.augment(coords)
        angles = 2 * math.pi * coords[..., None] / self.periods  # [..., N, 2, D/4]
        angles = angles.flatten(-2)  # [..., N, D/2]
        angles = torch.cat([angles, angles], dim=-1)  # [..., N, D]
        return torch.sin(angles), torch.cos(angles)


def pad_rope(rope: Rope, num_prefix: int) -> Rope:
    """Prepend identity rotations (sin=0, cos=1) for `num_prefix` tokens."""
    sin, cos = rope
    shape = (*sin.shape[:-2], num_prefix, sin.shape[-1])
    return (
        torch.cat([sin.new_zeros(shape), sin], dim=-2),
        torch.cat([cos.new_ones(shape), cos], dim=-2),
    )


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dim: [x1, x2] -> [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, rope: Rope) -> Tensor:
    """Apply RoPE to x of shape [B, heads, N, head_dim].

    sin/cos have shape [N, head_dim] or [B, N, head_dim].
    """
    sin, cos = rope
    if sin.ndim == 3:  # noqa: PLR2004
        sin, cos = sin[:, None], cos[:, None]
    sin, cos = sin.to(x.dtype), cos.to(x.dtype)
    return x * cos + rotate_half(x) * sin


def build_norm(norm: Literal["layernorm", "rmsnorm"], dim: int) -> nn.Module:
    """Build normalization layer."""
    match norm:
        case "layernorm":
            return nn.LayerNorm(dim, eps=1e-6)
        case "rmsnorm":
            return nn.RMSNorm(dim, eps=1e-6)
        case _:
            msg = f"Invalid norm {norm}. Expected 'layernorm' or 'rmsnorm'."
            raise ValueError(msg)


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
        rope: Rope | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply attention.

        Args:
            x: [B, N, C] tokens.
            rope: (sin, cos) of shape [N, head_dim] or [B, N, head_dim],
                identity on prefix tokens.
            attn_mask: [B, N] bool key padding mask, True = token is attended.
                Each sample must keep at least one token (e.g. its CLS).

        """
        B, N, C = x.shape  # noqa: N806
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, heads, N, head_dim]
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
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
        rope: Rope | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), rope, attn_mask)))
        return x + self.drop_path(self.ls2(self.ffn(self.norm2(x))))


class Transformer(nn.Module):
    """Stack of blocks, without final norm (owned by the backbone).

    `drop_path` is either a float (uniform rate) or a list with one rate per
    block, so that a backbone split into several stacks can apply a single
    linear drop path schedule across all of them.
    """

    def __init__(  # noqa: PLR0913
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
        self.apply(init_weights)

    def forward(
        self,
        x: Tensor,
        rope: Rope | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        for block in self.blocks:
            x = block(x, rope, attn_mask)
        return x


def init_weights(module: nn.Module) -> None:
    """Truncated normal init for linear layers, zero biases."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

