"""Utils module."""

import torch
import torch.nn.functional as F  # noqa: N812
from matplotlib import cm
from matplotlib.colors import to_hex
from torch import Tensor
from torchvision.utils import draw_segmentation_masks

pastis_cm = [
    [0, 0, 0],
    [0.6823529411764706, 0.7803921568627451, 0.9098039215686274],
    [1.0, 0.4980392156862745, 0.054901960784313725],
    [1.0, 0.7333333333333333, 0.47058823529411764],
    [0.17254901960784313, 0.6274509803921569, 0.17254901960784313],
    [0.596078431372549, 0.8745098039215686, 0.5411764705882353],
    [0.8392156862745098, 0.15294117647058825, 0.1568627450980392],
    [1.0, 0.596078431372549, 0.5882352941176471],
    [0.5803921568627451, 0.403921568627451, 0.7411764705882353],
    [0.7725490196078432, 0.6901960784313725, 0.8352941176470589],
    [0.5490196078431373, 0.33725490196078434, 0.29411764705882354],
    [0.7686274509803922, 0.611764705882353, 0.5803921568627451],
    [0.8901960784313725, 0.4666666666666667, 0.7607843137254902],
    [0.9686274509803922, 0.7137254901960784, 0.8235294117647058],
    [0.4980392156862745, 0.4980392156862745, 0.4980392156862745],
    [0.7803921568627451, 0.7803921568627451, 0.7803921568627451],
    [0.7372549019607844, 0.7411764705882353, 0.13333333333333333],
    [0.8588235294117647, 0.8588235294117647, 0.5529411764705883],
    [0.09019607843137255, 0.7450980392156863, 0.8117647058823529],
    [1, 1, 1]
]


def get_segment_mask_from_logits(logits: Tensor, num_classes: int) -> Tensor:
    """Convert prediction logits to mask for segmentation task."""
    return (
        F.one_hot(
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
    return (torch.sigmoid(logits) > 0.5).long()[0, 0].type(torch.bool)  # noqa: PLR2004


def get_target_mask_from_batch(
    batch_target: Tensor,
    num_classes: int,
    missing_val: int,
) -> Tensor:
    """Convert batch target mask to one hot mask."""
    batch_target = torch.where(batch_target == missing_val, num_classes, batch_target)
    target_msk = F.one_hot(
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
            #cm.get_cmap("plasma", num_classes).colors,
            pastis_cm,
        ),
    )
    msk = msk.unsqueeze(0)
    msk = F.interpolate(msk.float(), img.shape[-2:], mode="nearest").bool()
    msk = msk.squeeze(0)
    return draw_segmentation_masks(img, msk, 1, colormap)


def resize_bonus_mod(img: Tensor, bonus_img: Tensor) -> Tensor:
    bonus_img = bonus_img.unsqueeze(0)
    bonus_img = F.interpolate(bonus_img, img.shape[-2:], mode="nearest")
    bonus_img = bonus_img.squeeze(0)
    return bonus_img


def process_pastis_img(img: Tensor) -> Tensor:
    img = img[[2, 1, 0], :, :]
    mx = img.amax(dim=(1, 2))
    mi = img.amin(dim=(1, 2))
    img = (img - mi[:, None, None]) / (mx - mi)[:, None, None]
    img = torch.clip(img, 0, 1)
    return img