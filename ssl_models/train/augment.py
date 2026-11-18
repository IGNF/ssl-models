"""Geo-aligned multi-modal augmentation, on GPU, shared by all models.

All modalities of a sample cover the same extent. For each view, a single box is
sampled in normalized coordinates and every modality is cropped with that box
(`affine_grid` + `grid_sample`), then resampled to the view size of the
modality (image size for global views, DINO local size for local views).
Without crop, modalities are simply resized. Flips and transposition are drawn
per sample and view, and shared by all modalities, dates and raster targets of
the view. Elevation is then rescaled, and brightness/contrast jitter and blur are
applied to the inputs.

The n views of a type are made by repeating each sample n times (sample-major,
`out[k].unflatten(0, (B, n))`), then augmenting all rows at once: every random
draw is per row, hence per view. Global views keep the keys of the batch, local
views (if any) are at `{key}_local`.

Crops, jitter and blur are only applied when `random=True` (train stage), flips
always (if enabled).
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.dino.dino import dino_patch_size, local_view_size

REF_BLUR_SIZE = 224  # image size of reference for the blur sigma range


class Augment:
    """Single or multi-view augmentation of multi-modal, multi-temporal batches."""

    def __init__(  # noqa: PLR0913
        self,
        dataset: DatasetConfig,
        interpolate: str = "bilinear",
        flip: bool = True,
        ratio: Sequence[float] = (3 / 4, 4 / 3),
        jitter_prob: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        blur_sigma: Sequence[float] = (0.1, 2.0),
        blur_min_size: int = 32,
        n_global: int = 1,
        global_scale: Sequence[float] = (1.0, 1.0),
        global_blur_probs: Sequence[float] = (0.0,),
        n_local: int = 0,
        local_scale: Sequence[float] = (0.05, 0.32),
        local_blur_probs: Sequence[float] = (0.0,),
        local_size_ratio: float = 0.4375,
    ) -> None:
        """Set up augmentation, with `Augment(dataset, **vars(augment_config))`.

        The defaults give a single view of the full extent (resize only), with
        flips: parameters not in a config (e.g. `AugmentConfig`) keep these.

        Args:
            dataset: dataset config (view sizes, elevation rescaling, targets).
            interpolate: interpolation mode of the inputs.
            flip: random flips and transposition.
            ratio: aspect ratio range of the crop boxes.
            jitter_prob: brightness/contrast jitter probability.
            brightness: brightness jitter amplitude (in band std).
            contrast: contrast jitter amplitude.
            blur_sigma: blur sigma range, for a view size of `REF_BLUR_SIZE`.
            blur_min_size: no blur below this view size.
            n_global: number of global views.
            global_scale: area fraction range of the global crops ([1, 1]: no crop).
            global_blur_probs: blur probability of each global view (the last value
                is used for the remaining views).
            n_local: number of local views.
            local_scale: area fraction range of the local crops.
            local_blur_probs: blur probability of each local view.
            local_size_ratio: local view size / image size, for the modalities
                without `local_size` (rounded to a multiple of the DINO patch size,
                see `local_view_size`).

        """
        self.dataset = dataset
        self.interpolate = interpolate
        self.flip = flip
        self.ratio = tuple(ratio)
        self.jitter_prob = jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.blur_sigma = tuple(blur_sigma)
        self.blur_min_size = blur_min_size
        self.n_global = n_global
        self.global_scale = tuple(global_scale)
        self.global_blur_probs = tuple(global_blur_probs)
        self.n_local = n_local
        self.local_scale = tuple(local_scale)
        self.local_blur_probs = tuple(local_blur_probs)
        self.local_size_ratio = local_size_ratio
        self.sizes = {
            "global": {
                name_mod: mod.image_size for name_mod, mod in dataset.inputs.items()
            },
            "local": (
                {
                    name_mod: mod.local_size
                    or local_view_size(
                        mod.image_size, dino_patch_size(mod), local_size_ratio
                    )
                    for name_mod, mod in dataset.inputs.items()
                }
                if n_local
                else {}
            ),
        }

    def sample_boxes(
        self,
        batch_size: int,
        scale: tuple[float, float],
        device: torch.device,
    ) -> Tensor:
        """Sample boxes as [B, 2, 3] affine matrices (output -> input coords)."""
        area = torch.empty(batch_size, device=device).uniform_(*scale)
        log_ratio = torch.empty(batch_size, device=device).uniform_(
            math.log(self.ratio[0]), math.log(self.ratio[1])
        )
        fw = (area * log_ratio.exp()).sqrt().clamp(max=1.0)
        fh = (area / log_ratio.exp()).sqrt().clamp(max=1.0)
        cx = (2 * torch.rand(batch_size, device=device) - 1) * (1 - fw)
        cy = (2 * torch.rand(batch_size, device=device) - 1) * (1 - fh)

        theta = torch.zeros(batch_size, 2, 3, device=device)
        theta[:, 0, 0], theta[:, 0, 2] = fw, cx
        theta[:, 1, 1], theta[:, 1, 2] = fh, cy
        return theta

    @staticmethod
    def crop(
        x: Tensor,
        theta: Tensor | None,
        size: tuple[int, int],
        mode: str,
    ) -> Tensor:
        """Crop [B, T, C, H, W] with boxes theta (None: full extent) to size."""
        B, T, C = x.shape[:3]  # noqa: N806
        x = x.flatten(1, 2)
        if theta is None:
            if tuple(x.shape[-2:]) != tuple(size):
                x = F.interpolate(x, size=size, mode=mode)
            return x.unflatten(1, (T, C))

        grid = F.affine_grid(theta, (B, T * C, *size), align_corners=False)
        x = F.grid_sample(
            x.float(),
            grid,
            mode=mode,
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

    @staticmethod
    def rescale_elev(x: Tensor) -> Tensor:
        """Rescale elevation bands as differences with the first band."""
        return torch.cat([x[:, :, :1], 30 * (x[:, :, :1] - x[:, :, 1:])], dim=2)

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

    def blur(self, x: Tensor, prob: Tensor | float) -> Tensor:
        """Gaussian blur with per-sample sigma, applied with probability prob.

        `prob` is a float or a per-sample [B] tensor.
        """
        B, T, C, H, W = x.shape  # noqa: N806
        device = x.device
        blur_sigma = self.blur_sigma
        sigma_fac = H / REF_BLUR_SIZE
        sigma = torch.empty(B, device=device).uniform_(*blur_sigma) * sigma_fac
        active = torch.rand(B, device=device) < prob
        if not active.any():
            return x

        half = min(math.ceil(3 * blur_sigma[1] * sigma_fac), (H - 1) // 2)
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

    def make_views(  # noqa: PLR0913
        self,
        batch: dict[str, Tensor],
        n: int,
        scale: tuple[float, float],
        blur_probs: tuple[float, ...],
        is_global: bool,
        random: bool,
    ) -> dict[str, Tensor]:
        """Make n views per sample, stacked sample-major ([B * n, ...]).

        All entries are first repeated n times per sample, then each row is
        augmented as one view: modalities resized/cropped to the view sizes, raster
        targets transformed like the inputs (at their native size), other entries
        (dates, labels, ...) left as repeated.

        Args:
            batch: batch [B, ...].
            n: number of views per sample.
            scale: area fraction range of the crop boxes ([1, 1]: no crop).
            blur_probs: blur probability of view i: `blur_probs[i]`, the last value
                being used for the remaining views.
            is_global: global views (keys of the batch, image sizes) or local
                views (keys `{key}_local`, local sizes).
            random: apply random crops, jitter and blur.

        """
        suffix = "" if is_global else "_local"
        sizes = self.sizes["global"] if is_global else self.sizes["local"]

        x_ref = batch[next(iter(sizes))]
        B, device = x_ref.shape[0], x_ref.device  # noqa: N806
        if n > 1:
            batch = {k: v.repeat_interleave(n, dim=0) for k, v in batch.items()}
        blur_prob = torch.tensor(
            [blur_probs[min(idx, len(blur_probs) - 1)] for idx in range(n)],
            device=device,
        ).repeat(B)  # [B * n], view-dependent

        theta = (
            self.sample_boxes(B * n, scale, device=device)
            if random and scale != (1.0, 1.0)
            else None
        )
        flips = (
            torch.rand(B * n, 3, device=device) < 0.5  # noqa: PLR2004
            if self.flip
            else None
        )
        blur = random and bool(blur_prob.any())

        views = dict(batch)  # other entries (dates, ...) as repeated
        for name_mod, mod in self.dataset.inputs.items():
            x = self.crop(
                batch[name_mod],
                theta,
                (sizes[name_mod], sizes[name_mod]),
                self.interpolate,
            )
            if flips is not None:
                x = self.apply_flips(x, flips)
            if mod.rescale_elev:
                x = self.rescale_elev(x)
            if random and self.jitter_prob:
                x = self.jitter(x)
            if blur and sizes[name_mod] >= self.blur_min_size:
                x = self.blur(x, blur_prob)
            views[name_mod] = x

        for name_target in self.dataset.targets:
            if name_target not in self.dataset.rasters:
                continue
            y = batch[name_target]
            if theta is not None:
                y = self.crop(y, theta, tuple(y.shape[-2:]), "nearest").to(y.dtype)
            if flips is not None:
                y = self.apply_flips(y, flips)
            views[name_target] = y
        return {f"{key}{suffix}": value for key, value in views.items()}

    def __call__(
        self,
        batch: dict[str, Tensor],
        random: bool = True,
    ) -> dict[str, Tensor]:
        """Augment a batch into global (and local) views.

        Args:
            batch: modalities [B, T, C, H, W], all covering the same extent, and
                other entries (dates, targets, ...) with a leading batch dim.
            random: apply random crops, jitter and blur.

        Returns:
            Batch with the same keys as the input one, holding the `n_global` global
            views stacked sample-major (`out[k].unflatten(0, (B, n_global))`):
            modalities [B * n_global, T, C, size, size] at the image sizes, raster
            targets transformed, other entries repeated. With `n_local` > 0, the
            local views are added likewise at `{key}_local` for all entries
            (modalities at the local sizes, e.g. `{name_mod}_dates_local`), stacked
            sample-major ([B * n_local, ...]). With a single global view and
            no local view (default), a batch like the input one.

        """
        views = self.make_views(
            batch,
            self.n_global,
            self.global_scale,
            self.global_blur_probs,
            is_global=True,
            random=random,
        )
        if self.n_local > 0:
            local = self.make_views(
                batch,
                self.n_local,
                self.local_scale,
                self.local_blur_probs,
                is_global=False,
                random=random,
            )
            views.update(local)
        return views
