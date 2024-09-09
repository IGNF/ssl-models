"""Code adapted from https://github.com/Sense-X/MixMIM/tree/master.

# References:
# Swin: https://github.com/microsoft/Swin-Transformer
# timm: https://github.com/rwightman/pytorch-image-models
"""

from math import ceil
from typing import Callable

import torch
from timm.models.layers import trunc_normal_
from timm.models.swin_transformer import PatchMerging
from timm.models.vision_transformer import Block
from torch import Tensor, nn

from ssl_models.layers.swin import MixMAELayer
from ssl_models.ssl.mim import BaseMIM


class MixMAE(BaseMIM):
    """Mixed ans Masked Auto Encoder (MixMAE)."""

    def __init__(  # noqa: PLR0913
        self,
        in_chans: int,
        image_size: int,
        patch_size: int,
        out_patch_size: int,
        unpool: bool,
        num_levels: int,
        embed_dim: int,
        depths: tuple,
        num_heads: tuple,
        window_size: tuple,
        swin_version: int,
        v2_attn: bool,
        nb_mix_images: int = 4,
        type_head: str = "linear",
        loss_fn: Callable = torch.square,
        norm_pix_loss: bool = True,
        fac_abs_enc: float = 1.0,
        decoder_depth: int = 2,
        decoder_dim: int = 512,
        decoder_num_heads: int = 16,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pretrained_window_sizes: tuple[int, int, int, int] = [0, 0, 0, 0],
        use_checkpoint: bool = False,
        **kwargs,  # noqa: ANN003, ARG002
    ) -> None:
        super().__init__(
            in_chans=in_chans,
            image_size=image_size,
            patch_size=patch_size,
            out_patch_size=out_patch_size,
            unpool=unpool,
            num_levels=num_levels,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            type_head=type_head,
            loss_fn=loss_fn,
            norm_pix_loss=norm_pix_loss,
            fac_abs_enc=fac_abs_enc,
        )
        self.nb_mix_images = nb_mix_images
        self.patch_grid = (image_size // patch_size,) * 2
        self.out_patch_grid = (image_size // out_patch_size,) * 2

        # encoder args
        encoder_dim = embed_dim * 2 ** (num_levels - 1)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # encoder layers
        self.encoder_layers = nn.ModuleList()
        for idx_layer in range(num_levels):
            self.encoder_layers.append(
                MixMAELayer(
                    dim=embed_dim * 2**idx_layer,
                    input_resolution=(
                        self.patch_grid[0] // (2**idx_layer),
                        self.patch_grid[1] // (2**idx_layer),
                    ),
                    depth=depths[idx_layer],
                    num_heads=num_heads[idx_layer],
                    window_size=window_size[idx_layer],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:idx_layer]) : sum(depths[: idx_layer + 1])
                    ],
                    norm_layer=nn.LayerNorm,
                    downsample=(PatchMerging if (idx_layer < num_levels - 1) else None),
                    use_checkpoint=use_checkpoint,
                    pretrained_window_size=pretrained_window_sizes[idx_layer],
                    swin_version=swin_version,
                    nb_mix_images=nb_mix_images,
                    v2_attn=v2_attn,
                ),
            )
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(decoder_depth)
            ],
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        if swin_version == 2:  # noqa: PLR2004
            for bly in self.encoder_layers:
                bly.init_respostnorm()

    def mask_or_mix(self, x: Tensor):  # noqa: ANN201
        """Mix sequence."""
        H, W = self.patch_grid  # noqa: N806
        L = self.out_patch_grid[0] * self.out_patch_grid[1]  # noqa: N806
        B, _, C = x.shape  # noqa: N806
        B_enc = B // self.nb_mix_images  # noqa: N806
        if B % self.nb_mix_images:
            msg = f"Invalid batch size {B}. Expected multiple of {self.nb_mix_images}"
            raise ValueError(msg)

        # sample masked tokens
        noise = torch.randn((B_enc, L), device=x.device)  # [B L]
        random_indices = torch.argsort(noise, dim=-1)  # [B L]

        mask = torch.arange(self.nb_mix_images, dtype=x.dtype, device=x.device)
        mask = mask[None, :].repeat((B_enc, ceil(L / self.nb_mix_images)))[:, :L]
        mask = torch.gather(
            mask,
            dim=1,
            index=random_indices,
        )
        mask_rec = (
            mask[:, None, :]
            == torch.arange(self.nb_mix_images, device=x.device)[None, :, None]
        )
        mask_rec = 1 - mask_rec.reshape((B, L, 1)).float()

        mask = mask.reshape((B_enc, 1, *self.out_patch_grid))
        mask_s1 = torch.nn.functional.interpolate(
            mask,
            size=(H, W),
            mode="nearest",
        )
        mask_s2 = torch.nn.functional.interpolate(
            mask,
            size=(H // 2, W // 2),
            mode="nearest",
        )
        mask_s3 = torch.nn.functional.interpolate(
            mask,
            size=(H // 4, W // 4),
            mode="nearest",
        )
        mask_s4 = torch.nn.functional.interpolate(
            mask,
            size=(H // 8, W // 8),
            mode="nearest",
        )

        mask_s1 = mask_s1.reshape((B_enc, -1)).contiguous()
        mask_s2 = mask_s2.reshape((B_enc, -1)).contiguous()
        mask_s3 = mask_s3.reshape((B_enc, -1)).contiguous()
        mask_s4 = mask_s4.reshape((B_enc, -1)).contiguous()

        # encode only unmasked tokens
        mask_enc = (
            mask_s1[:, None, :]
            == torch.arange(self.nb_mix_images, device=x.device)[None, :, None]
        )
        mask_enc = mask_enc.reshape((B_enc, self.nb_mix_images, -1, 1)).float()
        x_enc = x.reshape((B_enc, self.nb_mix_images, -1, C))
        x_enc = (x_enc * mask_enc).sum(dim=1)

        return x_enc, mask_s1, mask_s2, mask_s3, mask_s4, mask_rec

    def unmask_or_unmix(
        self,
        x: Tensor,
        mask_rec: Tensor,
    ) -> Tensor:
        """Unmix sequence."""
        B_enc, L, C = x.shape  # noqa: N806

        # collate encoded patches and mask tokens
        mask_rec = mask_rec.reshape((B_enc, self.nb_mix_images, L, 1))
        x = x.reshape((B_enc, 1, L, C))
        x_dec = torch.where(mask_rec == 0, x, self.mask_token)
        return x_dec.flatten(0, 1)

    def encode(
        self,
        x: Tensor,
        mask_s1: Tensor | None,
        mask_s2: Tensor | None,
        mask_s3: Tensor | None,
        mask_s4: Tensor | None,
    ) -> Tensor:
        """Apply MixMAE encoder."""
        for idx, layer in enumerate(self.encoder_layers):
            if idx == 0:
                x = layer(x, attn_mask=mask_s1)
            elif idx == 1:
                x = layer(x, attn_mask=mask_s2)
            elif idx == 2:  # noqa: PLR2004
                x = layer(x, attn_mask=mask_s3)
            elif idx == 3:  # noqa: PLR2004
                x = layer(x, attn_mask=mask_s4)
        return self.encoder_norm(x)

    def encoder_to_decoder(self, x: Tensor) -> Tensor:
        """Apply MixMAE encoder-to-decoder."""
        return self.enc_to_dec(x)

    def decode(self, x: Tensor) -> Tensor:
        """Apply MixMAE decoder."""
        for _, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        return self.decoder_norm(x)


def mixmae_tiny(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE tiny."""
    match kwargs["num_levels"]:
        case 3:
            args = {
                "embed_dim": 48,
                "depths": [2, 2, 8],
                "num_heads": [2, 4, 8],
                "window_size": [16, 16, 16],
                "decoder_depth": 1,
            }
        case 4:
            args = {
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": [16, 16, 16, 8],
                "decoder_depth": 1,
            }
    args.update(kwargs)
    return MixMAE(**args)


def mixmae_small(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE small."""
    match kwargs["num_levels"]:
        case 3:
            args = {
                "embed_dim": 96,
                "depths": [2, 2, 8],
                "num_heads": [3, 6, 12],
                "window_size": [16, 16, 16],
                "decoder_depth": 2,
            }
        case 4:
            args = {
                "embed_dim": 96,
                "depths": [2, 2, 18, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": [16, 16, 16, 8],
                "decoder_depth": 2,
            }
    args.update(kwargs)
    return MixMAE(**args)


def mixmae_base(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE base."""
    match kwargs["num_levels"]:
        case 3:
            args = {
                "embed_dim": 192,
                "depths": [2, 2, 8],
                "num_heads": [6, 12, 24],
                "window_size": [16, 16, 16],
                "decoder_depth": 3,
            }
        case 4:
            args = {
                "embed_dim": 128,
                "depths": [2, 2, 18, 2],
                "num_heads": [4, 8, 16, 32],
                "window_size": [16, 16, 16, 8],
                "decoder_depth": 3,
            }
    args.update(kwargs)
    return MixMAE(**args)


def mixmae_large(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE large."""
    match kwargs["num_levels"]:
        case 3:
            args = {
                "embed_dim": 192,
                "depths": [2, 2, 20],
                "num_heads": [6, 12, 24],
                "window_size": [16, 16, 16],
                "decoder_depth": 4,
            }
        case 4:
            args = {
                "embed_dim": 192,
                "depths": [2, 2, 18, 2],
                "num_heads": [6, 12, 24, 48],
                "window_size": [16, 16, 16, 8],
                "decoder_depth": 4,
            }
    args.update(kwargs)
    return MixMAE(**args)


def mixmae_huge(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE huge."""
    match kwargs["num_levels"]:
        case 3:
            args = {
                "embed_dim": 352,
                "depths": [2, 2, 20],
                "num_heads": [11, 22, 44],
                "window_size": [16, 16, 16],
                "decoder_depth": 4,
            }
        case 4:
            args = {
                "embed_dim": 352,
                "depths": [2, 2, 18, 2],
                "num_heads": [11, 22, 44, 88],
                "window_size": [16, 16, 16, 8],
                "decoder_depth": 4,
            }
    args.update(kwargs)
    return MixMAE(**args)
