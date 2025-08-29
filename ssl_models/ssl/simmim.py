"""SimMIM module."""

from typing import Literal

import torch
from torch import Tensor, nn
from vit_pytorch.vit import Transformer

from conf.datasets import DatasetsConfig
from conf.mask import MaskConfig
from ssl_models.ssl.mim import BaseMIM


class SimMIM(BaseMIM):
    """SimMIM."""

    def __init__(  # noqa: PLR0913
        self,
        datasets: DatasetsConfig,
        mask: MaskConfig,
        multimodal: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        allmods_depth: tuple[int],
        model: Literal["simmim"],
        num_levels: Literal[1],
        unpool_dim: int | None,
        embed_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        type_head: Literal["linear", "attentive"] = "linear",
        loss_fn: Literal[torch.abs, torch.square] = torch.square,
        norm_pix_loss: bool = True,
        fac_abs_enc: float = 1.0,
        fac_date_enc: float = 1.0,
        **kwargs,  # noqa: ANN003, ARG002
    ) -> None:
        super().__init__(
            datasets=datasets,
            multimodal=multimodal,
            model=model,
            num_levels=num_levels,
            unpool_dim=unpool_dim,
            embed_dim=embed_dim,
            decoder_dim=embed_dim,
            type_head=type_head,
            loss_fn=loss_fn,
            norm_pix_loss=norm_pix_loss,
            fac_abs_enc=fac_abs_enc,
            fac_date_enc=fac_date_enc,
        )

        self.len_bands = {
            name_mod: 1 if isinstance(mod.bands, int) else len(mod.bands)
            for name_mod, mod in datasets.dataset.inputs.items()
        }

        num_dates_mod, num_dates_group = {}, {}
        for name_mod, name_group in datasets.dataset.groups:
            mod = datasets.dataset.inputs[name_mod]
            num_dates = mod.num_dates * self.len_bands[name_mod]
            if name_mod not in num_dates_mod:
                num_dates_mod[name_mod] = 0
            if name_group not in num_dates_group:
                num_dates_group[name_group] = 0
            num_dates_mod[name_mod] += num_dates
            num_dates_group[name_group] += num_dates

        (
            self.mask_ratio,
            self.mask_mod,
            self.mask_bands,
            self.mask_dates,
            self.mask_loc,
        ) = (
            {},
            {},
            {},
            {},
            {},
        )
        match multimodal:
            case "msgfm" | "shared" | "monotemp":
                name_models = (
                    list(num_dates_mod.keys())
                    if multimodal == "monotemp"
                    else ["shared"]
                )
                for name_mod in num_dates_mod:
                    self.mask_ratio[name_mod] = mask.mask_ratio
                    self.mask_mod[name_mod] = None
                    self.mask_bands[name_mod] = None
                    self.mask_dates[name_mod] = None
                    self.mask_loc[name_mod] = None
            case "mod" | "group":
                name_models = (
                    list(num_dates_group.keys())
                    if multimodal == "group"
                    else list(num_dates_mod.keys())
                )
                for name_mod, name_group in datasets.dataset.groups:
                    mod = datasets.dataset.inputs[name_mod]
                    if multimodal == "group":
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
                msg = f"Invalid multimodal mode {multimodal}."
                raise ValueError(msg)

        self.fake_cloud_val = {}
        for name_mod, mod in datasets.dataset.inputs.items():
            if not mod.fake_cloud_val:
                continue
            self.fake_cloud_val[name_mod] = mod.fake_cloud_val
            if self.mask_dates[name_mod] <= 0:
                msg = "Use of cloud masks requires temporally-structured masking."
                raise ValueError(msg)

        self.encoder = nn.ModuleDict(
            {
                name_mod: Transformer(
                    dim=embed_dim,
                    depth=depth - allmods_depth,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=embed_dim * mlp_ratio,
                )
                for name_mod in name_models
            },
        )
        if allmods_depth:
            self.encoder_all = Transformer(
                dim=embed_dim,
                depth=allmods_depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=embed_dim * mlp_ratio,
            )
        else:
            self.encoder_all = None

    def mask_struct(
        self,
        x: dict[str, Tensor],
        mask: dict[str, Tensor],
        stage: Literal["train", "val", "test"],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Structural masking of modalities, band groups, dates."""
        if not self.fake_cloud_val:
            return self.mask_struct_random(x)

        mask = self.ungroup(mask)

        mask_rec, mask_loss = {}, {}
        for name_mod in mask:
            if name_mod not in self.fake_cloud_val:
                mask_rec[name_mod] = mask[name_mod].bool()
                mask_loss[name_mod] = mask[name_mod].bool()
            # elif stage == "test":
            #     mask_rec[name_mod] = mask[name_mod].bool()
            #     mask_loss[name_mod] = mask[name_mod] == self.fake_cloud_val[name_mod]
            else:
                B, D, L, _ = mask[name_mod].shape  # noqa: N806
                G = self.len_bands[name_mod]  # noqa: N806

                mask[name_mod] = mask[name_mod].bool()
                mask_loss[name_mod] = torch.zeros(
                    (B, D, L, 1),
                    dtype=torch.bool,
                    device=mask[name_mod].device,
                )
                while not (
                    mask_loss[name_mod].any(dim=(1, 2)) | mask[name_mod].all(dim=(1, 2))
                ).all():
                    mask_loss[name_mod] = torch.where(
                        mask_loss[name_mod].any(dim=(1, 2), keepdim=True),
                        mask_loss[name_mod],
                        torch.rand(
                            (B, D // G, 1, 1),
                            device=mask[name_mod].device,
                        ).repeat(
                            1,
                            G,
                            1,
                            1,
                        )
                        < self.mask_dates[name_mod],
                    )
                    mask_loss[name_mod] = mask_loss[name_mod] & ~mask[name_mod]

                mask_rec[name_mod] = mask_loss[name_mod] | mask[name_mod]

        return self.group(mask_rec), self.group(mask_loss)

    def mask_seq(
        self,
        x: Tensor,
        mask_token: Tensor,
        mask_rec: Tensor,
        mask_loss: Tensor,
        encoding: Tensor,
        name_group: str,
    ) -> tuple[Tensor, Tensor, None, Tensor, Tensor]:
        """Mask sequence."""
        B, L, _ = x.shape  # noqa: N806
        if not self.fake_cloud_val:
            # sample masked tokens
            noise = torch.rand((B, L), device=x.device)  # [B L]
            noise *= 1 - mask_rec.reshape((B, L)).float()
            random_indices = torch.argsort(noise, dim=-1)  # [B L]
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
            mask_loss = mask_rec

        x_enc = torch.where(
            ~mask_rec,
            x,
            mask_token + encoding,
        )  # [B L:(1 - mask_ratio) C]
        return x_enc, mask_token, None, mask_rec, mask_loss

    def unmask_seq(
        self,
        x: Tensor,
        mask_token: Tensor,  # noqa: ARG002
        mask_rec: Tensor,  # noqa: ARG002
    ) -> Tensor:
        """SimMIM does not use unmasking."""
        return x

    def encode(
        self,
        x: dict[str, Tensor],
        mask_attn: dict[str, None],  # noqa: ARG002
    ) -> dict[str, Tensor]:
        """Apply SimMIM encoder."""
        x = self.encode_or_decode(x, model=self.encoder)
        if self.encoder_all:
            x = self.encode_or_decode_all(x, model=self.encoder_all)
        return x

    def encoder_to_decoder(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """SimMIM does not use encoder-to-decoder."""
        return x

    def dec_add_encodings(
        self,
        x_dec: dict[str, Tensor],
        dates: dict[str, Tensor],  # noqa: ARG002
        ref_date: Tensor,  # noqa: ARG002
    ) -> dict[str, Tensor]:
        """SimMIM does not use positional/date encodings before decoder."""
        return x_dec

    def decode(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """SimMIM does not use decoder."""
        return x


def simmim_tiny(**kwargs) -> SimMIM:  # noqa: ANN003
    """Construct SimMIM tiny."""
    args = {
        "embed_dim": 192,
        "depth": 12,
        "heads": 3,
        "dim_head": 64,
        "mlp_ratio": 2,
    }
    args.update(kwargs)
    return SimMIM(**args)


def simmim_small(**kwargs) -> SimMIM:  # noqa: ANN003
    """Construct SimMIM small."""
    args = {
        "embed_dim": 384,
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "mlp_ratio": 2,
    }
    args.update(kwargs)
    return SimMIM(**args)


def simmim_medium(**kwargs) -> SimMIM:  # noqa: ANN003
    """Construct SimMIM base/medium."""
    args = {
        "embed_dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
    }
    args.update(kwargs)
    return SimMIM(**args)


def simmim_large(**kwargs) -> SimMIM:  # noqa: ANN003
    """Construct SimMIM large."""
    args = {
        "embed_dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
    }
    args.update(kwargs)
    return SimMIM(**args)
