"""Geo-aligned multi-modal multi-crop augmentation, on GPU.

All modalities of a sample cover the same extent. For each view, a single box
is sampled in normalized coordinates and every modality is cropped with that
box (`affine_grid` + `grid_sample`), then resampled to the view size of the
modality. Flips, transposition, brightness/contrast jitter and blur are drawn
per sample and view, and are shared by all modalities and dates of the view.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

REF_BLUR_SIZE = 224  # image size of reference for the blur sigma range


class MultiCropAugment:
    """Multi-crop augmentation for multi-modal, multi-temporal batches."""

    def __init__(  # noqa: PLR0913
        self,
        n_global: int = 2,
        n_local: int = 8,
        global_scale: tuple[float, float] = (0.32, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.32),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        flip: bool = True,
        jitter_prob: float = 0.8,
        brightness: float = 0.2,
        contrast: float = 0.2,
        blur_probs: tuple[float, float, float] = (1.0, 0.1, 0.5),
        blur_sigma: tuple[float, float] = (0.1, 2.0),
        blur_min_size: int = 32,
    ) -> None:
        """Set up augmentation.

        Args:
            n_global: number of global views.
            n_local: number of local views.
            global_scale: area fraction range of global boxes.
            local_scale: area fraction range of local boxes.
            ratio: aspect ratio range of boxes.
            flip: random horizontal/vertical flips and transposition.
            jitter_prob: probability of brightness/contrast jitter.
            brightness: max brightness shift, in units of per-band std.
            contrast: max relative contrast change.
            blur_probs: blur probability for the first global view, the other
                global views and the local views.
            blur_sigma: blur sigma range, in pixels of a 224 px image.
            blur_min_size: modalities with smaller view size are not blurred.

        """
        self.n_global = n_global
        self.n_local = n_local
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        self.flip = flip
        self.jitter_prob = jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.blur_probs = blur_probs
        self.blur_sigma = blur_sigma
        self.blur_min_size = blur_min_size

    def sample_boxes(
        self,
        batch_size: int,
        scale: tuple[float, float],
        device: torch.device,
    ) -> Tensor:
        """Sample boxes as [B, 2, 3] affine matrices (output -> input coords)."""
        area = torch.empty(batch_size, device=device).uniform_(*scale)
        log_ratio = torch.empty(batch_size, device=device).uniform_(*self.log_ratio)
        fw = (area * log_ratio.exp()).sqrt().clamp(max=1.0)
        fh = (area / log_ratio.exp()).sqrt().clamp(max=1.0)
        cx = (2 * torch.rand(batch_size, device=device) - 1) * (1 - fw)
        cy = (2 * torch.rand(batch_size, device=device) - 1) * (1 - fh)

        theta = torch.zeros(batch_size, 2, 3, device=device)
        theta[:, 0, 0], theta[:, 0, 2] = fw, cx
        theta[:, 1, 1], theta[:, 1, 2] = fh, cy
        return theta

    @staticmethod
    def crop(x: Tensor, theta: Tensor, size: int) -> Tensor:
        """Crop [B, T, C, H, W] with boxes theta, resampled to size x size."""
        B, T, C = x.shape[:3]  # noqa: N806
        x = x.flatten(1, 2).float()
        grid = F.affine_grid(theta, (B, T * C, size, size), align_corners=False)
        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return x.unflatten(1, (T, C))

    @staticmethod
    def apply_flips(x: Tensor, flips: Tensor) -> Tensor:
        """Apply per-sample flips [B, 3] (h-flip, v-flip, transpose)."""
        x = torch.where(flips[:, 0, None, None, None, None], x.flip(-1), x)
        x = torch.where(flips[:, 1, None, None, None, None], x.flip(-2), x)
        return torch.where(
            flips[:, 2, None, None, None, None],
            x.transpose(-1, -2),
            x,
        )

    def jitter(self, x: Tensor) -> Tensor:
        """Brightness/contrast jitter per sample and band, shared across dates."""
        B, _, C = x.shape[:3]  # noqa: N806
        device = x.device
        active = (torch.rand(B, 1, device=device) < self.jitter_prob).float()
        contrast = torch.empty(B, C, device=device).uniform_(
            -self.contrast,
            self.contrast,
        )
        brightness = torch.empty(B, C, device=device).uniform_(
            -self.brightness,
            self.brightness,
        )
        contrast = (active * contrast)[:, None, :, None, None]
        brightness = (active * brightness)[:, None, :, None, None]

        mean = x.mean(dim=(1, 3, 4), keepdim=True)
        std = x.std(dim=(1, 3, 4), keepdim=True)
        return mean + (x - mean) * (1 + contrast) + brightness * std

    def blur(self, x: Tensor, prob: float) -> Tensor:
        """Gaussian blur with per-sample sigma, applied with probability prob."""
        B, T, C, H, W = x.shape  # noqa: N806
        device = x.device
        sigma_fac = H / REF_BLUR_SIZE
        sigma = torch.empty(B, device=device).uniform_(*self.blur_sigma) * sigma_fac
        active = torch.rand(B, device=device) < prob
        if not active.any():
            return x

        half = min(math.ceil(3 * self.blur_sigma[1] * sigma_fac), (H - 1) // 2)
        if half < 1:
            return x
        pos = torch.arange(-half, half + 1, device=device, dtype=x.dtype)
        kernel = torch.exp(-(pos[None] ** 2) / (2 * sigma[:, None] ** 2))
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        identity = (pos == 0).to(x.dtype).expand(B, -1)
        kernel = torch.where(active[:, None], kernel, identity)  # [B, k]
        kernel = kernel.repeat_interleave(T * C, dim=0)  # [B * T * C, k]

        y = x.reshape(1, B * T * C, H, W)
        y = F.pad(y, (half, half, 0, 0), mode="reflect")
        y = F.conv2d(y, kernel[:, None, None, :], groups=B * T * C)
        y = F.pad(y, (0, 0, half, half), mode="reflect")
        y = F.conv2d(y, kernel[:, None, :, None], groups=B * T * C)
        return y.reshape(B, T, C, H, W)

    def make_view(
        self,
        batch: dict[str, Tensor],
        sizes: dict[str, int],
        scale: tuple[float, float],
        blur_prob: float,
    ) -> dict[str, Tensor]:
        """Make one view of all modalities."""
        x_ref = batch[next(iter(sizes))]
        B, device = x_ref.shape[0], x_ref.device  # noqa: N806
        theta = self.sample_boxes(B, scale, device)
        flips = (
            torch.rand(B, 3, device=device) < 0.5  # noqa: PLR2004
            if self.flip
            else torch.zeros(B, 3, dtype=torch.bool, device=device)
        )

        view = {}
        for name_mod, size in sizes.items():
            x = self.crop(batch[name_mod], theta, size)
            x = self.apply_flips(x, flips)
            if self.jitter_prob:
                x = self.jitter(x)
            if blur_prob and size >= self.blur_min_size:
                x = self.blur(x, blur_prob)
            view[name_mod] = x
        return view

    def __call__(
        self,
        batch: dict[str, Tensor],
        sizes: dict[str, dict[str, int]],
    ) -> dict[str, list[dict[str, Tensor]]]:
        """Make global and local views.

        Args:
            batch: modalities [B, T, C, H, W], all covering the same extent.
            sizes: {"global" | "local": {modality: view size}}.

        Returns:
            {"global": [view, ...], "local": [view, ...]}, each view being a
            dict of modalities [B, T, C, size, size].

        """
        device_type = batch[next(iter(sizes["global"]))].device.type
        views = {"global": [], "local": []}
        # augmentations in float32, whatever the training precision
        with torch.autocast(device_type=device_type, enabled=False):
            for idx in range(self.n_global):
                blur_prob = self.blur_probs[0] if idx == 0 else self.blur_probs[1]
                views["global"].append(
                    self.make_view(
                        batch,
                        sizes["global"],
                        self.global_scale,
                        blur_prob,
                    ),
                )
            for _ in range(self.n_local):
                views["local"].append(
                    self.make_view(
                        batch,
                        sizes["local"],
                        self.local_scale,
                        self.blur_probs[2],
                    ),
                )
        return views
