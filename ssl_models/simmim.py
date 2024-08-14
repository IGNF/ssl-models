"""Code from https://github.com/microsoft/SimMIM.

# References:
# Swin: https://github.com/microsoft/Swin-Transformer
# timm: https://github.com/rwightman/pytorch-image-models
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------
"""

import torch
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.swin_transformer_v2 import SwinTransformerV2
from torch import Tensor, nn

from ssl_models.head import AttentiveHead, LinearHead
from ssl_models.odeon_data import OdeonData


class SwinTransformerForSimMIM(SwinTransformerV2):
    def __init__(self, drop_rate: float, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        if self.num_classes != 0:
            msg = (
                f"Invalid num classes {self.num_classes}.",
                "Expected num classes equals to 0.",
            )
            raise ValueError(msg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, self.embed_dim),
        )  ## TODO check avec mixmae
        trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.patch_embed(x)

        B, H, W, C = x.shape  # noqa: N806
        L = H * W  # noqa: N806
        x = x.view(B, L, C)

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)

        x = x * (1.0 - w) + mask_tokens * w

        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = x.view(
            B,
            H,
            W,
            C,
        )

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        B, H, W, C = x.shape  # noqa: N806
        return rearrange(x, "b h w c -> b c h w", b=B, h=H, w=W, c=C)


class SimMIM(nn.Module):
    def __init__(
        self,
        norm_pix_loss: bool,
        in_chans: int,
        encoder: SwinTransformerForSimMIM,
        encoder_stride: int,
        embed_dim: int,
        depths: tuple[int, int, int, int],
        image_size: int,
        patch_size: int,
        mask_patch_size: int,
        mask_ratio: float,
        type_head: str = "attentive",
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.norm_pix_loss = norm_pix_loss
        self.in_chans = in_chans
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio

        num_layers = len(depths)
        self.num_classes = OdeonData.NUM_CLASSES
        self.head = nn.ModuleDict(
            [
                [
                    dataset,
                    (
                        AttentiveHead(
                            embed_dim * 2 ** (num_layers - 1),
                            num_classes,
                        )
                        if type_head == "attentive"
                        else LinearHead(
                            embed_dim * 2 ** (num_layers - 1),
                            num_classes,
                        )
                    ),
                ]
                for dataset, num_classes in self.num_classes.items()
            ],
        )

    def random_masking(self, x: Tensor, ssl_phase: str) -> Tensor:
        if self.image_size % self.patch_size != 0:
            msg = (
                f"Invalid patch size {self.patch_size}."
                f"Expected divisor of image size {self.image_size} ",
            )
            raise ValueError(msg)
        if self.mask_patch_size % self.patch_size != 0:
            msg = (
                f"Invalid patch size {self.patch_size}."
                f"Expected divisor of mask patch size {self.mask_patch_size} ",
            )
            raise ValueError(msg)

        B, _, _, _ = x.shape  # noqa: N806
        rand_size = self.image_size // self.mask_patch_size
        scale = self.mask_patch_size // self.patch_size

        token_count = rand_size**2
        mask_count = (token_count * self.mask_ratio).__ceil__()

        if ssl_phase == "pretrain":
            mask_list = []
            for i in range(B):  # noqa: B007
                mask_idx = torch.randperm(token_count)[:mask_count]
                mask = torch.zeros((token_count), dtype=int, device=x.device)
                mask[mask_idx] = 1
                mask_list.append(mask)
            mask = torch.stack(mask_list, dim=0)
        else:
            mask = torch.zeros((B, token_count), dtype=int, device=x.device)

        mask = mask.reshape((B, rand_size, rand_size))
        return mask.repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)

    def forward_loss(self, x: Tensor, x_rec: Tensor, mask: Tensor) -> Tensor:
        if self.norm_pix_loss:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.0e-6) ** 0.5

        loss_rec = (x - x_rec) ** 2

        loss = torch.masked_select(loss_rec, mask.bool())

        return loss.mean()

    def forward(
        self,
        x: Tensor,
        ssl_phase: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        B, C, H, W = x["image"].shape  # noqa: N806
        mask = self.random_masking(x["image"], ssl_phase)
        z = self.encoder(x["image"], mask)
        x_rec = self.decoder(z)

        if ssl_phase == "pretrain":
            mask = (
                mask.repeat_interleave(self.patch_size, 1)
                .repeat_interleave(self.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )

            loss = self.forward_loss(x["image"], x_rec, mask)
            masked = torch.where(mask == 0, x["image"], 1)
            pred = torch.where(mask == 0, x["image"], x_rec)

            return loss, masked, pred, None

        z = rearrange(
            z,
            "b c h w -> b (h w) c",
            b=B,
            h=H // self.encoder_stride,
            w=W // self.encoder_stride,
        )

        head_logits = {
            dataset: self.head[dataset](z, ssl_phase) for dataset in self.num_classes
        }

        return None, None, None, head_logits


def simmim_tiny(**kwargs) -> SimMIM:  # noqa: ANN003
    """Construct SimMIM tiny."""
    args = {
        "in_chans": 3,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
    }
    encoder_args = {
        "in_chans": 3,
        "num_classes": 0,
        "num_heads": [3, 6, 12, 24],
        "window_size": 8,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "img_size": kwargs["image_size"],
    }
    encoder_args.update(kwargs)
    encoder_args.update(args)
    simmim_args = {"encoder_stride": 32, "mask_patch_size": 32}
    simmim_args.update(args)
    simmim_args.update(kwargs)
    return SimMIM(encoder=SwinTransformerForSimMIM(**encoder_args), **simmim_args)


def simmim_small(**kwargs) -> SimMIM:  # noqa: ANN003
    """Construct SimMIM small."""
    args = {
        "in_chans": 3,
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
    }
    encoder_args = {
        "in_chans": 3,
        "num_classes": 0,
        "num_heads": [3, 6, 12, 24],
        "window_size": 8,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "img_size": kwargs["image_size"],
    }
    encoder_args.update(kwargs)
    encoder_args.update(args)
    simmim_args = {"encoder_stride": 32, "mask_patch_size": 32}
    simmim_args.update(args)
    simmim_args.update(kwargs)
    return SimMIM(encoder=SwinTransformerForSimMIM(**encoder_args), **simmim_args)


def simmim_base(**kwargs) -> SimMIM:  # noqa: ANN003
    """Construct SimMIM base."""
    args = {
        "in_chans": 3,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
    }
    encoder_args = {
        "in_chans": 3,
        "num_classes": 0,
        "num_heads": [4, 8, 16, 32],
        "window_size": 8,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "img_size": kwargs["image_size"],
    }
    encoder_args.update(kwargs)
    encoder_args.update(args)
    simmim_args = {"encoder_stride": 32, "mask_patch_size": 32}
    simmim_args.update(args)
    simmim_args.update(kwargs)
    return SimMIM(encoder=SwinTransformerForSimMIM(**encoder_args), **simmim_args)


def simmim_large(**kwargs) -> SimMIM:  # noqa: ANN003
    """Construct SimMIM large."""
    args = {
        "in_chans": 3,
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
    }
    encoder_args = {
        "in_chans": 3,
        "num_classes": 0,
        "num_heads": [6, 12, 24, 48],
        "window_size": 8,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "img_size": kwargs["image_size"],
    }
    encoder_args.update(kwargs)
    encoder_args.update(args)
    simmim_args = {"encoder_stride": 32, "mask_patch_size": 32}
    simmim_args.update(args)
    simmim_args.update(kwargs)
    return SimMIM(encoder=SwinTransformerForSimMIM(**encoder_args), **simmim_args)


def simmim_huge(**kwargs) -> SimMIM:  # noqa: ANN003
    """Construct SimMIM huge."""
    args = {
        "in_chans": 3,
        "embed_dim": 352,
        "depths": [2, 2, 18, 2],
    }
    encoder_args = {
        "in_chans": 3,
        "num_classes": 0,
        "num_heads": [11, 22, 44, 88],
        "window_size": 8,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "img_size": kwargs["image_size"],
    }
    encoder_args.update(kwargs)
    encoder_args.update(args)
    simmim_args = {"encoder_stride": 32, "mask_patch_size": 32}
    simmim_args.update(args)
    simmim_args.update(kwargs)
    return SimMIM(encoder=SwinTransformerForSimMIM(**encoder_args), **simmim_args)
