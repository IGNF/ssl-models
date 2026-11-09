"""iBOT token masks and view-level modality/date dropout.

Both are generated on GPU and vectorized over the batch.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def ibot_masks(  # noqa: PLR0913
    token_shapes: dict[str, tuple[int, int, int]],
    batch_size: int,
    mask_prob: float,
    mask_ratio: tuple[float, float],
    blob_grid: int,
    shared: float,
    device: torch.device,
) -> dict[str, Tensor]:
    """Sample iBOT masks, as blobs aligned in space across modalities.

    A fraction `mask_prob` of samples is masked, with a ratio drawn uniformly in
    `mask_ratio` (the same ratio for all modalities and dates of a sample).
    Masks are obtained by thresholding smooth noise: a coarse `blob_grid` x
    `blob_grid` noise, bilinearly upsampled to each token grid. Since all grids
    cover the same extent, the noise shared by all sequences of a sample gives
    geographically aligned masks across modalities and dates. `shared` in
    [0, 1] mixes it with independent noise per (band group, date).

    Args:
        token_shapes: {modality: (S, h, w)}, S = band groups x dates.
        batch_size: number of samples.
        mask_prob: probability that a sample is masked.
        mask_ratio: (min, max) ratio of masked tokens.
        blob_grid: resolution of the coarse noise.
        shared: weight of the noise shared across modalities and dates.
        device: device of the masks.

    Returns:
        {modality: [batch_size, S, h * w]} bool masks, True = masked.

    """
    active = torch.rand(batch_size, device=device) < mask_prob
    ratio = torch.empty(batch_size, device=device).uniform_(*mask_ratio)
    ratio = ratio * active
    shared_noise = torch.rand(batch_size, 1, blob_grid, blob_grid, device=device)

    masks = {}
    for name_mod, (S, h, w) in token_shapes.items():  # noqa: N806
        noise_shared = F.interpolate(shared_noise, (h, w), mode="bilinear")
        noise_indep = torch.rand(
            batch_size * S,
            1,
            blob_grid,
            blob_grid,
            device=device,
        )
        noise_indep = F.interpolate(noise_indep, (h, w), mode="bilinear")
        noise = shared * noise_shared.reshape(batch_size, 1, h * w) + (
            1 - shared
        ) * noise_indep.reshape(batch_size, S, h * w)
        noise = noise + 1e-3 * torch.rand_like(noise)  # break ties

        num_masked = (ratio * h * w).round().long()
        ranks = noise.argsort(dim=-1, descending=True).argsort(dim=-1)
        masks[name_mod] = ranks < num_masked[:, None, None]
    return masks


def view_keep(
    num_dates: dict[str, int],
    batch_size: int,
    drop_mod: float | None,
    drop_dates: float | None,
    device: torch.device,
) -> dict[str, Tensor] | None:
    """Sample which (modality, date) are kept in a student view.

    Whole modalities are dropped with probability `drop_mod` (only if there are
    several modalities), single dates with probability `drop_dates` (only for
    multi-temporal modalities). At least one (modality, date) is kept per
    sample: if all were dropped, the sample keeps everything.

    Returns:
        {modality: [batch_size, num_dates]} bool, True = kept, or None if no
        dropout is configured.

    """
    if not drop_mod and not drop_dates:
        return None

    multi_mod = len(num_dates) > 1
    keep = {}
    for name_mod, T in num_dates.items():  # noqa: N806
        keep_mod = torch.ones(batch_size, T, dtype=torch.bool, device=device)
        if drop_mod and multi_mod:
            keep_mod &= torch.rand(batch_size, 1, device=device) >= drop_mod
        if drop_dates and T > 1:
            keep_mod &= torch.rand(batch_size, T, device=device) >= drop_dates
        keep[name_mod] = keep_mod

    any_kept = torch.stack([k.any(dim=1) for k in keep.values()], dim=1).any(dim=1)
    return {name_mod: k | ~any_kept[:, None] for name_mod, k in keep.items()}
