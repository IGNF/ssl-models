"""Utils module."""

import torch
from matplotlib import cm
from matplotlib.colors import to_hex
from torch import Tensor
from torch.nn.functional import one_hot
from torchvision.utils import draw_segmentation_masks

BANDS_RGB = 3


def get_pretrain_logs(
    batch: dict[str, Tensor],
    pixels_masked: dict[str, Tensor],
    pixels_rec: dict[str, Tensor],
    ssl_phase: str,
    stage: str,
    mods_to_plot: list[str],
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Format pretraining logs."""
    log_inputs = {
        f"{ssl_phase}_{stage}/_{name_mod}_input": img[0, 0]
        for name_mod, img in pixels_masked.items()
        if name_mod in mods_to_plot
    }
    log_preds = {
        f"{ssl_phase}_{stage}/_{name_mod}_rec": img[0, 0]
        for name_mod, img in pixels_rec.items()
        if name_mod in mods_to_plot
    }
    log_targets = {
        f"{ssl_phase}_{stage}/_{name_mod}_target": batch[name_mod][0, 0][:BANDS_RGB]
        for name_mod in pixels_masked
        if name_mod in mods_to_plot
    }
    return log_inputs, log_preds, log_targets


def get_segment_mask_from_logits(logits: Tensor, num_classes: int) -> Tensor:
    """Convert prediction logits to mask for segmentation task."""
    return (
        one_hot(
            torch.argmax(
                logits,
                dim=0,
            ),
            num_classes,
        )
        .type(torch.bool)
        .permute(2, 0, 1)
    )


def get_cd_mask_from_logits(logits: Tensor) -> Tensor:
    """Convert prediction logits to mask for change detection task."""
    return (
        (
            torch.sigmoid(logits) > 0.5  # noqa: PLR2004
        )
        .long()[0, 0]
        .type(torch.bool)
    )


def get_target_mask_from_batch(
    batch_target: Tensor,
    num_classes: int,
    missing_val: int,
) -> Tensor:
    """Convert batch target mask to one hot mask."""
    batch_target = torch.where(batch_target == missing_val, num_classes, batch_target)
    target_msk = one_hot(
        batch_target.type(torch.int64),
        num_classes + 1,
    )
    target_msk = target_msk[:, :, :num_classes]
    return target_msk.permute(2, 0, 1).type(torch.bool)


def create_masked_image(img: Tensor, msk: Tensor, num_classes: int) -> Tensor:
    """Create overlayed image with input image and mask."""
    colormap = list(
        map(
            to_hex,
            cm.get_cmap("plasma", num_classes).colors,
        ),
    )
    return draw_segmentation_masks(img, msk, 0.4, colormap)


def should_log_probe_finetune(type_target: str) -> bool:
    """Check whether images should be logged."""
    return type_target in ["segment", "change_detect"]
