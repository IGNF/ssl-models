"""Code adapted from https://github.com/Sense-X/MixMIM/tree/master.

# References:
# Swin: https://github.com/microsoft/Swin-Transformer
# timm: https://github.com/rwightman/pytorch-image-models
"""

from math import ceil
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from timm.models.swin_transformer import PatchMerging
from timm.models.vision_transformer import Block
from torch import Tensor, nn

from conf.datasets import DatasetsConfig
from ssl_models.layers.swin import MixMAELayer
from ssl_models.ssl.mim import BaseMIM


class MixMAE(BaseMIM):
    """Mixed ans Masked Auto Encoder (MixMAE)."""

    def __init__(  # noqa: PLR0913
        self,
        datasets: DatasetsConfig,
        model: Literal["mixmae"],
        unpool: bool,
        num_levels: Literal[3, 4],
        embed_dim: int,
        depths: tuple,
        num_heads: tuple,
        window_size: tuple,
        swin_version: Literal[1, 2],
        v2_attn: bool,
        nb_mix_images: int = 4,
        type_head: Literal["linear", "attentive"] = "linear",
        loss_fn: Literal[torch.square, torch.square] = torch.square,
        norm_pix_loss: bool = True,
        fac_abs_enc: float = 1.0,
        fac_date_enc: float = 1.0,
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
        multimodal: Literal["msgfm", "shared", "mod", "group"] = "mod",
        **kwargs,  # noqa: ANN003, ARG002
    ) -> None:
        super().__init__(
            datasets=datasets,
            model=model,
            unpool=unpool,
            num_levels=num_levels,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            type_head=type_head,
            loss_fn=loss_fn,
            norm_pix_loss=norm_pix_loss,
            fac_abs_enc=fac_abs_enc,
            fac_date_enc=fac_date_enc,
            multimodal=multimodal,
        )
        match multimodal:
            case "msgfm" | "shared":
                name_models = ["shared"]
            case "mod" | "group":
                msg = "Group SSL not yet implemented with MixMAE."
                raise NotImplementedError(msg)
            case _:
                msg = f"Invalid multimodal mode {multimodal}."
                raise ValueError(msg)

        self.nb_mix_images = nb_mix_images
        self.stride = 2 ** (num_levels - 1)

        # encoder args
        encoder_dim = embed_dim * 2 ** (num_levels - 1)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # encoder layers
        max_patch_grid = max(
            mod.image_size // mod.patch_size.mixmae
            for _, mod in datasets.dataset.inputs.items()
        )
        num_windows = [
            max(1, max_patch_grid // (window_size[idx_layer] * 2**idx_layer))
            for idx_layer, _ in enumerate(window_size)
        ]
        self.encoder_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        name_mod: MixMAELayer(
                            dim=embed_dim * 2**idx_layer,
                            depth=depths[idx_layer],
                            num_heads=num_heads[idx_layer],
                            num_windows=num_windows[idx_layer],
                            window_size=window_size[idx_layer],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[
                                sum(depths[:idx_layer]) : sum(depths[: idx_layer + 1])
                            ],
                            norm_layer=nn.LayerNorm,
                            downsample=(
                                PatchMerging if (idx_layer < num_levels - 1) else None
                            ),
                            use_checkpoint=use_checkpoint,
                            pretrained_window_size=pretrained_window_sizes[idx_layer],
                            swin_version=swin_version,
                            nb_mix_images=nb_mix_images,
                            v2_attn=v2_attn,
                        )
                        for name_mod in name_models
                    },
                )
                for idx_layer in range(num_levels)
            ],
        )
        self.encoder_norm = nn.ModuleDict(
            {name_mod: nn.LayerNorm(encoder_dim) for name_mod in name_models},
        )
        self.enc_to_dec = nn.ModuleDict(
            {name_mod: nn.Linear(encoder_dim, decoder_dim) for name_mod in name_models},
        )
        self.decoder_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        name_mod: Block(
                            decoder_dim,
                            decoder_num_heads,
                            mlp_ratio,
                            qkv_bias=True,
                            norm_layer=nn.LayerNorm,
                        )
                        for name_mod in name_models
                    },
                )
                for _ in range(decoder_depth)
            ],
        )
        self.decoder_norm = nn.ModuleDict(
            {name_mod: nn.LayerNorm(decoder_dim) for name_mod in name_models},
        )

    def mask_mod(
        self,
        x: Tensor,
        name_group: str,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Mix sequence."""
        B_out, L, C = x.shape  # noqa: N806
        H, W = round(L**0.5), round(L**0.5)  # noqa: N806
        H_out, W_out, L_out = (  # noqa: N806
            H // self.stride,
            W // self.stride,
            L // self.stride**2,
        )

        B = B_out // self.nb_mix_images  # noqa: N806
        if B_out % self.nb_mix_images:
            msg = (
                f"Invalid batch size {B_out}. Expected multiple of {self.nb_mix_images}"
            )
            raise ValueError(msg)

        # sample masked tokens
        noise = torch.randn((B, L_out), device=x.device)  # [B L_out]
        random_indices = torch.argsort(noise, dim=-1)  # [B L_out]

        mask = torch.arange(self.nb_mix_images, dtype=x.dtype, device=x.device)
        mask = mask[None, :].repeat((B, ceil(L_out / self.nb_mix_images)))[:, :L_out]
        mask = torch.gather(
            mask,
            dim=1,
            index=random_indices,
        )
        mask_rec = (
            mask[:, None, :]
            == torch.arange(self.nb_mix_images, device=x.device)[None, :, None]
        )
        mask_rec = 1 - mask_rec.reshape((B_out, L_out, 1)).float()

        mask = mask.reshape((B, 1, H_out, W_out))
        mask_attn = F.interpolate(
            mask,
            size=(H, W),
            mode="nearest",
        )
        mask_attn = mask_attn.reshape((B, L)).contiguous()

        # encode only unmasked tokens
        mask_enc = (
            mask_attn[:, None, :]
            == torch.arange(self.nb_mix_images, device=x.device)[None, :, None]
        )
        mask_enc = mask_enc.reshape((B, self.nb_mix_images, L, 1)).float()
        x_enc = x.reshape((B, self.nb_mix_images, L, C))
        x_enc = (x_enc * mask_enc).sum(dim=1)

        return x_enc, mask_attn, mask_rec

    def unmask_mod(self, x: Tensor, mask_rec: Tensor, mask_token: Tensor) -> Tensor:
        """Unmask sequence, filling with mask token."""
        B_enc, L, C = x.shape  # noqa: N806

        mask_rec = mask_rec.reshape((B_enc, self.nb_mix_images, L, 1))
        x = x.reshape((B_enc, 1, L, C))
        x_dec = torch.where(mask_rec == 0, x, mask_token)
        return x_dec.flatten(0, 1)

    def encode_with_mask(
        self,
        x: dict[str, Tensor],
        mask_attn: dict[str, Tensor | None],
        model: nn.Module,
    ) -> dict[str, Tensor]:
        """Apply encoder."""
        for name_group in x:
            model_group = model[name_group] if name_group in model else model["shared"]
            x[name_group], mask_attn[name_group] = model_group(
                x[name_group],
                mask_attn[name_group],
            )
        return x, mask_attn

    def encode(
        self,
        x: dict[str, Tensor],
        mask_attn: dict[str, Tensor | None],
    ) -> dict[str, Tensor]:
        """Apply MixMAE encoder."""
        for encoder_layer in self.encoder_layers:
            x, mask_attn = self.encode_with_mask(x, mask_attn, model=encoder_layer)
        return self.encode_or_decode(x, model=self.encoder_norm)

    def encoder_to_decoder(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply MixMAE encoder-to-decoder."""
        return self.encode_or_decode(x, model=self.enc_to_dec)

    def decode(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply MixMAE decoder."""
        for decoder_block in self.decoder_blocks:
            x = self.encode_or_decode(x, model=decoder_block)
        return self.encode_or_decode(x, model=self.decoder_norm)


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
