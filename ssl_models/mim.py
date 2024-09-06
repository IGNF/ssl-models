"""MIM base module, which is inherited in specialized MIM modules."""

from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from ssl_models.embed import Patchify, Pixelify
from ssl_models.head import AttentiveHead, LinearHead
from ssl_models.odeon_data import OdeonData
from ssl_models.utils import posemb_sincos_2d


class BaseMIM(nn.Module):
    """Masked Auto Encoder (MAE)."""

    def __init__(
        self,
        in_chans: int,
        image_size: int,
        patch_size: int,
        out_patch_size: int,
        unpool: bool,
        num_levels: int,
        embed_dim: int,
        decoder_dim: int,
        type_head: str = "linear",
        loss_fn: Callable = torch.square,
        norm_pix_loss: bool = True,
        fac_abs_enc: float = 1.0,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.out_patch_size = out_patch_size
        self.loss_fn = loss_fn
        self.norm_pix_loss = norm_pix_loss

        # patchify and pixelify
        self.patch_embed = Patchify(in_chans, embed_dim, patch_size, unpool)
        self.embed_to_rec = Pixelify(
            decoder_dim,
            in_chans,
            out_patch_size,
            unpool,
        )

        # Fix the position encodings to sine & cosine functions
        self.register_buffer(
            name="enc_pos_encoding",
            tensor=posemb_sincos_2d(
                h=image_size // patch_size,
                w=image_size // patch_size,
                dim=embed_dim,
            )
            .float()
            .unsqueeze(0)
            * fac_abs_enc,
            persistent=False,
        )
        self.register_buffer(
            name="dec_pos_encoding",
            tensor=posemb_sincos_2d(
                h=image_size // out_patch_size,
                w=image_size // out_patch_size,
                dim=decoder_dim,
            )
            .float()
            .unsqueeze(0),
            persistent=False,
        )
        # Freeze the weights of position encoding
        self.enc_pos_encoding = self.enc_pos_encoding.requires_grad_(
            requires_grad=False,
        )
        self.dec_pos_encoding = self.dec_pos_encoding.requires_grad_(
            requires_grad=False,
        )

        self.num_classes = OdeonData.NUM_CLASSES
        self.head = nn.ModuleDict(
            [
                [
                    dataset,
                    AttentiveHead(
                        embed_dim * 2 ** (num_levels - 1),
                        num_classes,
                    )
                    if type_head == "attentive"
                    else LinearHead(
                        embed_dim * 2 ** (num_levels - 1),
                        num_classes,
                    ),
                ]
                for dataset, num_classes in self.num_classes.items()
            ],
        )

    def compute_loss(
        self,
        target: Tensor,
        patches_rec: Tensor,
        mask_rec: Tensor,
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        patches_target = rearrange(
            target,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.out_patch_size,
            p2=self.out_patch_size,
        )

        if self.norm_pix_loss:
            mean = patches_target.mean(dim=-1, keepdim=True)
            var = patches_target.var(dim=-1, keepdim=True)
            patches_target = (patches_target - mean) / (var + 1.0e-6) ** 0.5

        loss_rec = self.loss_fn(patches_target - patches_rec)
        loss_rec = torch.masked_select(loss_rec, mask_rec.bool())

        return loss_rec.mean()

    def compute_pixels(
        self,
        target: Tensor,
        patches_rec: Tensor,
        mask_rec: Tensor,
    ) -> Tensor:
        """Shared visu of reconstruction images shown for all MIM modules."""
        mask_pixels = repeat(
            mask_rec,
            "b l 1 -> b l repeat",
            repeat=self.out_patch_size**2,
        )  # [B L PP]
        mask_pixels = rearrange(
            mask_pixels,
            "b (h w) (p1 p2) -> b 1 (h p1) (w p2)",
            h=self.image_size // self.out_patch_size,
            w=self.image_size // self.out_patch_size,
            p1=self.out_patch_size,
            p2=self.out_patch_size,
        )
        pixels_target = torch.where(mask_pixels == 1, 1, target)

        pixels_rec = rearrange(
            patches_rec,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.image_size // self.out_patch_size,
            w=self.image_size // self.out_patch_size,
            p1=self.out_patch_size,
            p2=self.out_patch_size,
        )
        pixels_rec = torch.where(mask_pixels == 1, pixels_rec, target)

        return pixels_target, pixels_rec

    def forward(
        self,
        image: Tensor,
        ssl_phase: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Shared forward pass for all MIM modules."""
        x_enc, embed_inds = self.patch_embed(image)
        x_enc = x_enc + self.enc_pos_encoding

        if ssl_phase == "pretrain":
            x_enc, mask_s1, mask_s2, mask_s3, mask_s4, mask_rec = self.mask_or_mix(
                x_enc,
            )
        else:
            mask_s1, mask_s2, mask_s3, mask_s4 = (None,) * 4

        x_enc = self.encode(
            x_enc,
            mask_s1,
            mask_s2,
            mask_s3,
            mask_s4,
        )

        if ssl_phase == "pretrain":
            x_dec = self.encoder_to_decoder(x_enc)
            x_dec = self.unmask_or_unmix(x_dec, mask_rec)

            x_dec = x_dec + self.dec_pos_encoding
            x_dec = self.decode(x_dec)

            patches_rec = self.embed_to_rec(
                x_dec,
                embed_inds,
            )

            rec_loss = self.compute_loss(
                image,
                patches_rec,
                mask_rec,
            )
            pixels_target, pixels_rec = self.compute_pixels(
                image,
                patches_rec,
                mask_rec,
            )
            return rec_loss, pixels_target, pixels_rec, None

        head_logits = {
            dataset: self.head[dataset](x_enc, ssl_phase)
            for dataset in self.num_classes
        }
        return None, None, None, head_logits
