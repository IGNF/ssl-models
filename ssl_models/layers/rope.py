"""Own implementation of 2D axial RoPE, with coordinate augmentations.

RoPE is computed from explicit patch coordinates normalized to [-1, 1] within
a view, so that tokens of different modalities covering the same location get
the same rotation, whatever their grid size. The CLS tokens (1 x 1 grid) are at
the center of the view, the registers get the identity rotation (see
`BaseSSL.add_registers`).
"""

import math

import torch
from torch import Tensor, nn


class Rope(nn.Module):
    """2D axial rotary position embedding from explicit coordinates.

    Half of the head dimension rotates with the y coordinate, the other half
    with x. At train time, coordinates can be augmented (`sample_augment`, with
    the same draw for all the modalities and CLS tokens of a view):
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

    @staticmethod
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

    def sample_augment(self) -> tuple[Tensor, Tensor] | None:
        """Sample the augmentation of a view: the shift [2] then the scale [2] of its
        (y, x) coordinates, None outside of training."""
        if not self.training:
            return None

        device = self.periods.device
        shift = torch.zeros(2, device=device)
        log_scale = torch.zeros(2, device=device)
        if self.shift_coords:
            shift.uniform_(-self.shift_coords, self.shift_coords)
        if self.jitter_coords:
            log_jitter = math.log(self.jitter_coords)
            log_scale += torch.empty(2, device=device).uniform_(-log_jitter, log_jitter)
        if self.rescale_coords:
            log_rescale = math.log(self.rescale_coords)
            log_scale += torch.empty(1, device=device).uniform_(
                -log_rescale,
                log_rescale,
            )
        return shift, log_scale.exp()

    def forward(
        self,
        batch_size: int,
        num_dates: int,
        grid_size: int,
        device: torch.device | None = None,
        augment: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """RoPE (sin, cos) [B, D, L, 2 * head_dim] of the L = grid_size x grid_size
        patches, the same for the B samples and D dates, with the coordinates
        augmented by `augment` (if any, see `sample_augment`)."""
        coords = self.grid_coords(grid_size, grid_size, device=device)
        coords = coords[None, None].expand(batch_size, num_dates, -1, -1)
        coords = coords.float()
        if augment is not None:
            shift, scale = augment
            coords = (coords + shift) * scale
        # [B, D, L, 2, head_dim / 4] -> [B, D, L, head_dim / 2]
        angles = 2 * math.pi * coords[..., None] / self.periods
        angles = angles.flatten(-2)
        return torch.cat(
            [
                torch.sin(angles),
                torch.sin(angles),
                torch.cos(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )  # [B, D, L, 2 * head_dim]


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dim: [x1, x2] -> [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    """Apply RoPE to x of shape [B, heads, L, head_dim].

    rope (sin, cos) has shape [L, 2 * head_dim] or [B, L, 2 * head_dim].
    """
    sin, cos = rope.chunk(2, dim=-1)
    if sin.ndim == 3:  # noqa: PLR2004
        sin, cos = sin[:, None], cos[:, None]
    sin, cos = sin.to(x.dtype), cos.to(x.dtype)
    return x * cos + rotate_half(x) * sin
