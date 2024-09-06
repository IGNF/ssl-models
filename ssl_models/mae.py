"""MAE module. Adapted from clay foundation model: https://github.com/Clay-foundation/model."""

from typing import Callable

import torch
from einops import rearrange
from torch import Tensor, nn
from vit_pytorch.vit import Transformer

from ssl_models.mim import BaseMIM


class MAE(BaseMIM):
    """Masked Auto Encoder (MAE)."""

    def __init__(  # noqa: PLR0913
        self,
        in_chans: int,
        image_size: int,
        patch_size: int,
        out_patch_size: int,
        unpool: bool,
        num_levels: int,
        embed_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
        decoder_dim_head: int,
        decoder_mlp_ratio: float,
        mask_ratio: float = 0.75,
        type_head: str = "linear",
        loss_fn: Callable = torch.square,
        norm_pix_loss: bool = True,
        fac_abs_enc: float = 1.0,
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
        self.mask_ratio = mask_ratio

        self.encoder = Transformer(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=embed_dim * mlp_ratio,
        )

        self.enc_to_dec = (
            nn.Linear(embed_dim, decoder_dim)
            if embed_dim != decoder_dim
            else nn.Identity()
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim))

        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=embed_dim * decoder_mlp_ratio,
        )

    def mask_or_mix(self, x: Tensor):  # noqa: ANN201
        """Mask sequence."""
        B, L, _ = x.shape  # noqa: N806

        # sample masked tokens
        noise = torch.randn((B, L), device=x.device)  # [B L]
        random_indices = torch.argsort(noise, dim=-1)  # [B L]
        reverse_indices = torch.argsort(random_indices, dim=-1)

        num_masked_patches = round(self.mask_ratio * L)  # Number of patches to mask
        mask_rec = torch.zeros((B, L), device=x.device)
        mask_rec[:, :num_masked_patches] = 1
        mask_rec = torch.gather(
            mask_rec,
            dim=1,
            index=reverse_indices,
        )
        mask_rec = mask_rec[:, :, None]

        # encode only unmasked tokens
        batch_indices = rearrange(torch.arange(B, device=x.device), "b -> b 1")
        unmasked_indices = random_indices[:, num_masked_patches:]
        unmasked_indices = unmasked_indices.sort(dim=1)
        x_enc = x[batch_indices, unmasked_indices, :]  # [B L:(1 - mask_ratio) C]
        return x_enc, None, None, None, None, mask_rec

    def unmask_or_unmix(self, x: Tensor, mask_rec: Tensor) -> Tensor:
        """Unmask sequence, filling with mask token."""
        B, L_enc, C = x.shape  # noqa: N806
        _, L, _ = mask_rec.shape  # noqa: N806

        # find masked and unmasked tokens
        mask_rec = mask_rec.squeeze(dim=2)
        mask_rec = mask_rec.argsort(dim=1, descending=True)
        masked_indices, unmasked_indices = torch.split(
            mask_rec,
            [L - L_enc, L_enc],
            dim=1,
        )
        unmasked_indices = unmasked_indices.sort(dim=1)

        # collate encoded patches and mask tokens
        batch_indices = rearrange(torch.arange(B, device=x.device), "b -> b 1")
        x_dec = torch.zeros((B, L, C), device=x.device)
        x_dec[batch_indices, masked_indices, :] = self.mask_token
        x_dec[batch_indices, unmasked_indices, :] = x
        return x_dec

    def encode(self, x: Tensor, *args) -> Tensor:  # noqa: ARG002, ANN002
        """Apply MAE encoder."""
        return self.encoder(x)

    def encoder_to_decoder(self, x: Tensor) -> Tensor:
        """Apply MAE encoder-to-decoder."""
        return self.enc_to_dec(x)

    def decode(self, x: Tensor) -> Tensor:
        """Apply MAE decoder."""
        return self.decoder(x)


def mae_tiny(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE tiny."""
    args = {
        "embed_dim": 192,
        "depth": 12,
        "heads": 3,
        "dim_head": 64,
        "mlp_ratio": 2,
        "decoder_dim": 512,
        "decoder_depth": 1,
        "decoder_heads": 16,
        "decoder_dim_head": 32,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_small(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE small."""
    args = {
        "embed_dim": 384,
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "mlp_ratio": 2,
        "decoder_dim": 512,
        "decoder_depth": 2,
        "decoder_heads": 16,
        "decoder_dim_head": 32,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_medium(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE base/medium."""
    args = {
        "embed_dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 3,
        "decoder_heads": 16,
        "decoder_dim_head": 32,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_large(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE large."""
    args = {
        "embed_dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 16,
        "decoder_dim_head": 32,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)
