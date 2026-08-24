"""Feature map extraction and visualization utilities."""

import torch
from torch import Tensor

RGB_BANDS = 3


def features_to_rgb_pca(
    features: Tensor,
    n_components: int = RGB_BANDS,
    spatial_dims: tuple[int, int] | None = None,
) -> Tensor:
    """Convert feature maps to RGB using PCA.

    Applies PCA to reduce feature dimension to 3, then normalizes to [0, 1].
    Optionally reshapes back to spatial dimensions.

    Parameters
    ----------
    features : Tensor
        Feature tensor of shape (B, L, D) or (B, N, D) where:
        - B: batch size
        - L or N: number of tokens/patches
        - D: feature dimension
    n_components : int, default=3
        Number of PCA components (typically 3 for RGB).
    spatial_dims : tuple[int, int], optional
        If provided, reshape output to (B, 3, H, W).

    Returns
    -------
    Tensor
        RGB visualization of shape (B, 3, H, W) if spatial_dims provided,
        else (B, L, 3) or (B, N, 3).
    """
    if features.dim() != 3:
        msg = f"Expected 3D features (B, L, D), got shape {features.shape}"
        raise ValueError(msg)

    batch_size, num_tokens, feat_dim = features.shape
    num_components = min(n_components, feat_dim, num_tokens)

    features_rgb_list = []
    for features_sample in features:
        # PCA backend kernels do not support float16 QR on CUDA.
        # Keep this test-time visualization path numerically stable on CPU float32.
        features_sample = features_sample.detach().to(dtype=torch.float32, device="cpu")

        # Center then project with low-rank PCA for stable/fast inference-time viz.
        sample_centered = features_sample - features_sample.mean(dim=0, keepdim=True)
        _, _, v = torch.pca_lowrank(sample_centered, q=num_components)
        sample_proj = sample_centered @ v[:, :num_components]

        if num_components < n_components:
            pad = torch.zeros(
                (num_tokens, n_components - num_components),
                device=sample_proj.device,
                dtype=sample_proj.dtype,
            )
            sample_proj = torch.cat((sample_proj, pad), dim=1)

        proj_min = sample_proj.min(dim=0, keepdim=True).values
        proj_max = sample_proj.max(dim=0, keepdim=True).values
        proj_range = (proj_max - proj_min).clamp_min(1e-6)
        sample_proj = (sample_proj - proj_min) / proj_range
        features_rgb_list.append(sample_proj)

    features_rgb = torch.stack(features_rgb_list, dim=0).float().cpu()

    # Reshape to spatial if requested
    if spatial_dims is not None:
        height, width = spatial_dims
        if num_tokens != height * width:
            msg = (
                f"Number of tokens {num_tokens} doesn't match spatial dims "
                f"{height}x{width}={height*width}"
            )
            raise ValueError(msg)
        features_rgb = features_rgb.permute(0, 2, 1)  # (B, 3, L)
        features_rgb = features_rgb.reshape(
            batch_size,
            3,
            height,
            width,
        )  # (B, 3, H, W)

    return features_rgb


def batch_pca_reduce(
    features_dict: dict[str, Tensor],
    spatial_dims: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Tensor]:
    """Apply PCA reduction to multiple feature tensors.

    Parameters
    ----------
    features_dict : dict[str, Tensor]
        Dictionary mapping modality names to feature tensors.
    spatial_dims : dict[str, tuple[int, int]], optional
        Dictionary mapping modality names to (H, W) spatial dimensions.

    Returns
    -------
    dict[str, Tensor]
        Dictionary of RGB-ized feature maps, same keys as input.
    """
    rgb_dict = {}
    for modality, features in features_dict.items():
        if features is None or features.numel() == 0:
            continue

        if features.dim() == 4:
            # Already spatial (B, C, H, W) - flatten to (B, L, C)
            B, C, H, W = features.shape
            features_2d = features.reshape(B, C, H * W).permute(0, 2, 1)
            spatial_dims_mod = (
                (H, W)
                if spatial_dims is None
                else spatial_dims.get(modality)
            )
        else:
            features_2d = features
            if spatial_dims is None:
                spatial_dims_mod = None
            else:
                spatial_dims_mod = spatial_dims.get(modality)

        rgb_dict[modality] = features_to_rgb_pca(
            features_2d,
            spatial_dims=spatial_dims_mod,
        )

    return rgb_dict