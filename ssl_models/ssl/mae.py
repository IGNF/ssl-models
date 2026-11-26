"""MAE module. Adapted from clay foundation model: https://github.com/Clay-foundation/model."""

from functools import partial
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.mask import MAEMaskConfig
from ssl_models.layers.embed import Pixelify
from ssl_models.layers.utils import shuffle_enc_to_dec
from ssl_models.layers.vit import Transformer
from ssl_models.ssl.base import BaseSSL


class MAE(BaseSSL):
    """Masked Auto Encoder (MAE), on top of the token pipeline of `BaseSSL`.

    The masked tokens are dropped before the encoder (`mask`), then the decoder
    reconstructs the pixels from the encoded tokens and the mask tokens (`unmask`).
    The encoder (one stack per model name, then an optional shared stack of
    `inter_depth` blocks) and the decoder are `Transformer` stacks. The encoder
    and decoder outputs are normalized, as well as the outputs of the stacks per
    model name before the shared stack if `inter_norm`. The defaults of the
    backbone options (absolute positions, no CLS tokens or registers, no qkv bias,
    no LayerScale, no drop path, PyTorch init) match the former `vit_pytorch`
    implementation, except for the decoder MLP, now sized from `decoder_dim`
    (formerly from `embed_dim`).
    """

    def __init__(  # noqa: PLR0913
        self,
        dataset: DatasetConfig,
        mask: MAEMaskConfig,
        fusion_mode: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        inter_depth: int,
        embed_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
        decoder_mlp_ratio: float,
        type_head: Literal["linear", "attentive"] = "attentive",
        fac_abs_enc: float = 1.0,
        fac_date_enc: float = 1.0,
        # backbone options
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        ffn: Literal["mlp", "swiglu"] = "mlp",
        layerscale_init: float | None = None,
        drop_path: float = 0.0,
        qkv_bias: bool = False,
        inter_norm: bool = True,
        trunc_normal_init: bool = False,
        patch_norm: bool = True,
        pos_enc: Literal["abs", "rope"] = "abs",
        use_cls: bool = False,
        num_registers: int = 0,
        rope_base: float = 100.0,
        rope_shift: float | None = None,
        rope_jitter: float | None = None,
        rope_rescale: float | None = None,
    ) -> None:
        # encoder stacks (uniform drop path rate over all blocks), and decoder
        # stacks with the same block options
        transformer = partial(
            Transformer,
            norm=norm,
            ffn=ffn,
            layerscale_init=layerscale_init,
            drop_path=drop_path,
            qkv_bias=qkv_bias,
            trunc_normal_init=trunc_normal_init,
        )
        super().__init__(
            dataset=dataset,
            fusion_mode=fusion_mode,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            patch_size={
                name_mod: mod.patch_size.mae for name_mod, mod in dataset.inputs.items()
            },
            transformer=partial(
                transformer,
                dim=embed_dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
            ),
            depth=depth,
            inter_depth=inter_depth,
            inter_norm=inter_norm,
            type_head=type_head,
            patch_norm=patch_norm,
            pos_enc=pos_enc,
            fac_abs_enc=fac_abs_enc,
            fac_date_enc=fac_date_enc,
            use_cls=use_cls,
            num_registers=num_registers,
            rope_head_dim=embed_dim // heads,
            decoder_rope_head_dim=decoder_dim // decoder_heads,
            rope_base=rope_base,
            rope_shift=rope_shift,
            rope_jitter=rope_jitter,
            rope_rescale=rope_rescale,
        )
        self.decoder_dim = decoder_dim
        self.shuffle_mods = fusion_mode == "msgfm"  # shuffling of modalities

        # masking parameters
        num_dates_mod, num_dates_group = {}, {}
        for name_mod, name_group in dataset.groups:
            num_dates = dataset.inputs[name_mod].num_dates * self.len_bands[name_mod]
            num_dates_mod[name_mod] = num_dates_mod.get(name_mod, 0) + num_dates
            num_dates_group[name_group] = num_dates_group.get(name_group, 0) + num_dates

        self.mask_ratio, self.mask_mod = {}, {}
        self.mask_bands, self.mask_dates, self.mask_loc = {}, {}, {}
        match fusion_mode:
            case "msgfm" | "shared" | "monotemp":
                for name_mod in num_dates_mod:
                    self.mask_ratio[name_mod] = mask.mask_ratio
                    self.mask_mod[name_mod] = None
                    self.mask_bands[name_mod] = None
                    self.mask_dates[name_mod] = None
                    self.mask_loc[name_mod] = None
            case "mod" | "group":
                for name_mod, name_group in dataset.groups:
                    mod = dataset.inputs[name_mod]
                    if fusion_mode == "group":
                        scale_fac = num_dates_group[name_group] ** mask.mask_scale
                        self.mask_ratio[name_group] = (
                            1 - (1 - mask.mask_ratio) / scale_fac
                        )
                        self.mask_mod[name_mod] = (
                            mask.mask_mod
                            if num_dates_mod[name_mod] != num_dates_group[name_group]
                            else None
                        )
                    else:
                        scale_fac = num_dates_mod[name_mod] ** mask.mask_scale
                        self.mask_ratio[name_mod] = (
                            1 - (1 - mask.mask_ratio) / scale_fac
                        )
                        self.mask_mod[name_mod] = None

                    self.mask_bands[name_mod] = (
                        mask.mask_bands if self.len_bands[name_mod] > 1 else None
                    )
                    self.mask_dates[name_mod] = (
                        mask.mask_dates if mod.num_dates > 1 else None
                    )
                    self.mask_loc[name_mod] = mask.mask_loc
            case _:
                msg = f"Invalid fusion mode {fusion_mode}."
                raise ValueError(msg)

        # decoder, with its mask tokens (its position encodings are set in `BaseSSL`)
        self.enc_to_dec = nn.ModuleDict(
            {
                name: (
                    nn.Linear(embed_dim, decoder_dim)
                    if embed_dim != decoder_dim
                    else nn.Identity()
                )
                for name in self.name_models
            },
        )
        self.decoder = nn.ModuleDict(
            {
                name: transformer(
                    dim=decoder_dim,
                    depth=decoder_depth,
                    heads=decoder_heads,
                    mlp_ratio=decoder_mlp_ratio,
                    drop_path=0.0,
                    final_norm=True,
                )
                for name in self.name_models
            },
        )
        self.mask_token = nn.ParameterDict(
            {
                name_mod: nn.Parameter(
                    torch.randn(1, len_bands, 1, 1, decoder_dim),
                )
                for name_mod, len_bands in self.len_bands.items()
            },
        )

        # pixel reconstruction, shared like the patch embeddings
        self.embed_to_rec = nn.ModuleDict()
        for name_mod, mod in dataset.inputs.items():
            name_embed = self.mod_embed[name_mod]
            if name_embed not in self.embed_to_rec:
                self.embed_to_rec[name_embed] = Pixelify(
                    decoder_dim,
                    mod.bands,
                    self.patch_embed[name_embed].patch_size,
                )

    def mask(
        self,
        x: dict[str, Tensor],
        rope: dict[str, Tensor] | None,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
    ]:
        """Mask the sequences in pretraining, keeping the visible tokens only.

        Returns the visible tokens and their RoPE, the decoder mask tokens
        {group: [B, L, C]}, and the reconstruction masks {group: [B, L, 1]} bool
        (True = masked). Outside of pretraining, all tokens are visible, and the last
        two are None.
        """
        if ssl_phase != "pretrain":
            return x, rope, None, None

        mask_token = {}
        for name_mod, x_mod in self.ungroup(x).items():
            B, _, L, _ = x_mod.shape
            D = self.dataset.inputs[name_mod].num_dates
            mask_token[name_mod] = (
                self.mask_token[name_mod]
                .expand((B, -1, D, L, -1))  # [B, G, D, L, C]
                .flatten(1, 2)
                .to(x_mod.dtype)
            )

        mask_token = self.group(mask_token)

        mask_rec = self.mask_struct(x)

        x_enc = {}
        for name_group in x:
            x_enc[name_group], mask_rec[name_group], batch_indices, unmasked_indices = (
                self.mask_seq(
                    x[name_group],
                    mask_rec[name_group],
                    name_group,
                )
            )
            if rope is not None:
                rope[name_group] = rope[name_group][batch_indices, unmasked_indices]

        return x_enc, rope, mask_token, mask_rec

    def mask_struct(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Structural masking of modalities, band groups, dates and locations.

        Returns {group: [B, L, 1]} bool masks (True = masked first by `mask_seq`).
        """
        mask_group, mask_mod = {}, {}
        for name_group in x:
            B, L, _ = x[name_group].shape  # noqa: N806
            mask_group[name_group] = torch.ones((B, L, 1), dtype=torch.bool)
            mask_mod[name_group] = torch.ones((B, L, 1), dtype=torch.bool)

        mask_mod = self.ungroup(mask_mod)

        while any(
            mask_group[name_group].all(dim=(1, 2)).any(dim=0)
            for name_group in mask_group
        ):
            mask = {}
            for name_mod, G in self.len_bands.items():
                B, _, L, _ = mask_mod[name_mod].shape
                D = self.dataset.inputs[name_mod].num_dates
                mask[name_mod] = torch.zeros((B, G, D, L, 1), dtype=torch.bool)
                if self.mask_mod[name_mod]:
                    mask[name_mod] += (
                        torch.rand((B, 1, 1, 1, 1)) < self.mask_mod[name_mod]
                    )
                if self.mask_bands[name_mod]:
                    mask[name_mod] += (
                        torch.rand((B, G, 1, 1, 1)) < self.mask_bands[name_mod]
                    )
                if self.mask_dates[name_mod]:
                    mask[name_mod] += (
                        torch.rand((B, 1, D, 1, 1)) < self.mask_dates[name_mod]
                    )
                if self.mask_loc[name_mod]:
                    mask[name_mod] += (
                        torch.rand((B, 1, 1, L, 1)) < self.mask_loc[name_mod]
                    )
                mask[name_mod] = mask[name_mod].flatten(1, 2)

            mask = self.group(mask)

            for name_group in mask_group:
                mask_group[name_group] = torch.where(
                    mask_group[name_group].all(dim=1, keepdim=True),
                    mask[name_group],
                    mask_group[name_group],
                )

        return {
            name_group: mask_group[name_group].to(x[name_group].device)
            for name_group in mask_group
        }

    def mask_seq(
        self,
        x: Tensor,
        mask_rec: Tensor,
        name_group: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Mask a sequence, keeping the visible tokens only.

        Returns the visible tokens, the reconstruction mask [B, L, 1] (True = masked),
        and the batch and token indices of the visible tokens.
        """
        B, L, _ = x.shape

        # sample masked tokens
        noise = torch.rand((B, L), device=x.device)  # [B, L]
        noise *= 1 - mask_rec.reshape((B, L)).float()
        random_indices = torch.argsort(noise, dim=-1)  # [B, L]
        reverse_indices = torch.argsort(random_indices, dim=-1)

        num_masked_patches = round(
            self.mask_ratio[name_group] * L,
        )  # Number of patches to mask
        mask_rec = torch.zeros((B, L), dtype=torch.bool, device=x.device)
        mask_rec[:, :num_masked_patches] = True
        mask_rec = torch.gather(
            mask_rec,
            dim=1,
            index=reverse_indices,
        )
        mask_rec = mask_rec[:, :, None]

        # encode only unmasked tokens
        batch_indices = rearrange(torch.arange(B, device=x.device), "b -> b 1")
        unmasked_indices, _ = random_indices[:, num_masked_patches:].sort(dim=1)
        x_enc = x[batch_indices, unmasked_indices, :]  # [B, (1 - mask_ratio) * L, C]

        return x_enc, mask_rec, batch_indices, unmasked_indices

    def unmask(
        self,
        x: dict[str, Tensor],
        mask_token: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Unmask the sequences: the encoded tokens x at the visible positions, the
        mask tokens at the masked ones."""
        x_dec = {}
        for name_group in x:
            x_dec[name_group] = (
                mask_token[name_group]
                .to(x[name_group].dtype)
                .masked_scatter(~mask_rec[name_group], x[name_group])
            )
        return x_dec

    def shuffle_enc_to_dec(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Shuffle modalities and dates."""
        if not self.shuffle_mods:
            return x

        x_dec = self.ungroup(x)

        x_dec = shuffle_enc_to_dec(x_dec)

        return self.group(x_dec)

    def encoder_to_decoder(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply MAE encoder-to-decoder."""
        x_dec = {}
        for name_group in x:
            model_group = (
                self.enc_to_dec[name_group]
                if name_group in self.enc_to_dec
                else self.enc_to_dec["shared"]
            )
            x_dec[name_group] = model_group(x[name_group])

        return x_dec

    def decode(
        self,
        x: dict[str, Tensor],
        rope: dict[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        """Apply MAE decoder (with RoPE, if any, all tokens attended)."""
        return self.encode_or_decode(x, rope, None, model=self.decoder)

    def rec_pixels(
        self,
        x_dec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Reconstruct pixels from embeddings."""
        x_dec = self.ungroup(x_dec)
        mask_rec = self.ungroup(mask_rec)

        pixels_rec = {}
        for name_mod in x_dec:
            name_embed = self.mod_embed[name_mod]
            pixels_rec[name_mod], mask_rec[name_mod] = self.embed_to_rec[name_embed](
                x_dec[name_mod],
                mask_rec[name_mod],
            )
        return pixels_rec, mask_rec

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> dict[str, dict[str, Tensor]]:
        """Forward pass for MAE module.

        Returns:
            In pretraining, "pixels_rec": the reconstructed pixels and
            "mask_rec": the reconstruction masks (True = masked pixel, on which the
            loss is computed), {modality: [B, D, C, H, W]}. In probing / finetuning,
            "logits": the logits of the heads.

        """
        x_enc, x_cls = self.embed(batch)
        x_enc, x_cls = self.add_enc_encodings(x_enc, x_cls, batch)
        rope_enc, rope_cls = self.add_enc_rope(x_enc, x_cls)

        x_enc, rope_enc, mask_token, mask_rec = self.mask(
            x_enc,
            rope_enc,
            ssl_phase=ssl_phase,
        )

        x_enc, x_cls = self.encode(x_enc, x_cls, rope_enc, rope_cls, None)

        if ssl_phase == "pretrain":
            x_dec = self.shuffle_enc_to_dec(x_enc)
            x_dec = self.encoder_to_decoder(x_dec)
            x_dec = self.unmask(x_dec, mask_token, mask_rec)
            x_dec = self.add_dec_encodings(x_dec, batch)
            rope_dec = self.add_dec_rope(x_dec)
            x_dec = self.decode(x_dec, rope_dec)
            pixels_rec, mask_rec = self.rec_pixels(x_dec, mask_rec)

            return {
                "mask_rec": mask_rec,
                "pixels_rec": pixels_rec,
            }
        elif ssl_phase in ("probe", "finetune"):
            logits = self.compute_logits(x_enc, x_cls, ssl_phase=ssl_phase)

            return {
                "logits": logits,
            }


def mae_tiny(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE tiny."""
    args = {
        "embed_dim": 192,
        "depth": 12,
        "heads": 3,
        "mlp_ratio": 2,
        "decoder_dim": 512,
        "decoder_depth": 1,
        "decoder_heads": 16,
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
        "mlp_ratio": 2,
        "decoder_dim": 512,
        "decoder_depth": 2,
        "decoder_heads": 16,
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
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 3,
        "decoder_heads": 16,
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
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 16,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)
