"""Generic MIM base module, inherited in specialized MIM modules."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat
from torch import Tensor, nn
from torchmetrics import MeanMetric

from conf.dataset.utils import RasterConfig
from conf.datasets import DatasetsConfig
from ssl_models.layers.embed import Patchify, Pixelify
from ssl_models.layers.head import AttentiveHead, LinearHead, PixelifyHead
from ssl_models.layers.utils import (
    encode_dates,
    flatten_dates,
    posemb_sincos_2d,
    reshape_encoding,
    shuffle_enc_to_dec,
    unflatten_dates,
)
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric


class BaseMIM(nn.Module, ABC):
    """Masked Auto Encoder (MAE)."""

    def __init__(  # noqa: C901, PLR0913
        self,
        datasets: DatasetsConfig,
        model: Literal["mae", "mixmae"],
        unpool: bool,
        num_levels: Literal[1, 3, 4],
        embed_dim: int,
        decoder_dim: int,
        type_head: Literal["linear", "attentive"] = "linear",
        loss_fn: Literal[torch.square, torch.abs] = torch.square,
        norm_pix_loss: bool = True,
        fac_abs_enc: float = 1.0,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        shuffle_mods: bool = False,
        mask_indep: bool = True,
        mask_shared: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = datasets.dataset
        self.loss_fn = loss_fn
        self.norm_pix_loss = norm_pix_loss
        self.stride = 2 ** (num_levels - 1)

        # patchify and pixelify
        self.grid_size, self.out_grid_size = {}, {}
        self.patch_embed, self.embed_to_rec = nn.ModuleDict(), nn.ModuleDict()
        for name_mod, mod in self.dataset.inputs.items():
            num_bands = mod.bands if isinstance(mod.bands, int) else len(mod.bands)
            patch_size = mod.patch_size.mae if model == "mae" else mod.patch_size.mixmae
            self.grid_size[name_mod] = mod.image_size // patch_size
            self.out_grid_size[name_mod] = mod.image_size // (patch_size * self.stride)
            self.patch_embed[name_mod] = Patchify(
                num_bands,
                embed_dim,
                patch_size,
                unpool,
            )
            self.embed_to_rec[name_mod] = Pixelify(
                decoder_dim,
                num_bands,
                patch_size * self.stride,
                unpool,
            )

        # Fix the position encodings to sine & cosine functions
        max_grid_size = max(self.grid_size.values())
        self.register_buffer(
            name="enc_pos_encoding",
            tensor=posemb_sincos_2d(
                h=max_grid_size,
                w=max_grid_size,
                dim=embed_dim,
                date_dim=date_dim,
            ).float()
            * fac_abs_enc,
            persistent=False,
        )
        self.register_buffer(
            name="dec_pos_encoding",
            tensor=posemb_sincos_2d(
                h=max_grid_size,
                w=max_grid_size,
                dim=decoder_dim,
                date_dim=date_dim,
            ).float(),
            persistent=False,
        )
        # Freeze the weights of position encoding
        self.enc_pos_encoding = self.enc_pos_encoding.requires_grad_(
            requires_grad=False,
        )
        self.dec_pos_encoding = self.dec_pos_encoding.requires_grad_(
            requires_grad=False,
        )

        # functions to construct date encodings
        self.enc_date_encoding = {
            name_mod: partial(
                encode_dates,
                dim=embed_dim,
                date_dim=date_dim,
                fac_date_enc=fac_date_enc,
                grid_size=self.grid_size[name_mod],
            )
            for name_mod in self.dataset.inputs
        }
        self.dec_date_encoding = {
            name_mod: partial(
                encode_dates,
                dim=decoder_dim,
                date_dim=date_dim,
                fac_date_enc=fac_date_enc,
                grid_size=self.out_grid_size[name_mod],
            )
            for name_mod in self.dataset.inputs
        }

        # flattening/unflattening of date dimensions
        self.flatten_dates = partial(
            flatten_dates,
            dim=0 if mask_indep else 1,
        )
        self.unflatten_dates = partial(
            unflatten_dates,
            dim=0 if mask_indep else 1,
            num_dates={
                name_mod: mod.num_dates for name_mod, mod in self.dataset.inputs.items()
            },
        )

        # shuffling of modalities
        self.shuffle_enc_to_dec = shuffle_enc_to_dec if shuffle_mods else nn.Identity()

        # mask tokens
        self.mask_token = nn.ParameterDict(
            {
                name_mod: nn.Parameter(torch.randn(1, 1, decoder_dim))
                for name_mod in (
                    ("shared",) if mask_shared else datasets.dataset.inputs
                )
            },
        )

        # heads
        self.heads = nn.ModuleDict()
        for name_target, target in self.dataset.targets.items():
            if isinstance(target, RasterConfig):
                if self.dataset.ref_input is None:
                    msg = f"Ref input must be provided for raster target {name_target}"
                    raise ValueError(msg)
                target_image_size = round(
                    self.dataset.crop_meters / target.resolution_meters,
                )
                ref_grid_size = self.out_grid_size[self.dataset.ref_input]
                if target_image_size % ref_grid_size:
                    msg = (
                        f"Target image size {target_image_size} "
                        f"is not a multiple of ref input grid {ref_grid_size}"
                    )
                    raise ValueError(msg)
                self.heads[name_target] = PixelifyHead(
                    embed_dim * self.stride,
                    target.num_classes,
                    target_image_size // ref_grid_size,
                )
            elif type_head == "attentive":
                self.heads[name_target] = AttentiveHead(
                    embed_dim * self.stride,
                    target.num_classes,
                )
            else:
                self.heads[name_target] = LinearHead(
                    embed_dim * self.stride,
                    target.num_classes,
                )

        self.loss_pred = {}
        self.metrics = nn.ModuleDict()
        for name_target, target in datasets.dataset.targets.items():
            match target.type_target:
                case "classif" | "segment":
                    self.loss_pred[name_target] = F.cross_entropy
                    metric_partial = partial(
                        MonoLabelMetric,
                        type_target=target.type_target,
                        num_classes=target.num_classes,
                    )
                case "change_detect":
                    self.loss_pred[name_target] = F.binary_cross_entropy
                    metric_partial = partial(
                        MonoLabelMetric,
                        type_target=target.type_target,
                        num_classes=target.num_classes,
                    )
                case "multilabel_classif":
                    self.loss_pred[name_target] = F.binary_cross_entropy_with_logits
                    metric_partial = partial(
                        MultiLabelMetric,
                        num_labels=target.num_classes,
                    )
            for stage in ("train", "val"):
                self.metrics[f"{name_target}_{stage}"] = metric_partial()

        for name_loss in ("loss_rec", "loss_pred"):
            for stage in ("train", "val"):
                self.metrics[f"{name_loss}_{stage}"] = MeanMetric(
                    dist_sync_on_step=False,
                )

    def embed(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, list[Tensor]], dict[str, Tensor], Tensor]:
        """Embed patches and fetch dates."""
        x_enc, embed_inds, dates = {}, {}, {}
        for name_mod in self.dataset.inputs:
            x_enc[name_mod], embed_inds[name_mod] = self.patch_embed[name_mod](
                batch[name_mod],
            )
            dates[name_mod] = batch[f"{name_mod}_dates"]

        return self.flatten_dates(x_enc), embed_inds, dates, batch["ref_date"]

    def enc_add_encodings(
        self,
        x_enc: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
    ) -> dict[str, Tensor]:
        """Add positional and date encodings before encoder."""
        x_enc = self.unflatten_dates(x_enc)

        for name_mod in x_enc:
            pos_encoding = reshape_encoding(
                self.enc_pos_encoding,
                self.grid_size[name_mod],
            )
            date_encoding = self.enc_date_encoding[name_mod](
                dates=dates[name_mod],
                ref_date=ref_date,
            )
            x_enc[name_mod] += pos_encoding + date_encoding

        return self.flatten_dates(x_enc)

    def dec_add_encodings(
        self,
        x_dec: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
    ) -> dict[str, Tensor]:
        """Add positional and date encodings before decoder."""
        x_dec = self.unflatten_dates(x_dec)

        for name_mod in x_dec:
            pos_encoding = reshape_encoding(
                self.dec_pos_encoding,
                self.out_grid_size[name_mod],
            )
            date_encoding = self.dec_date_encoding[name_mod](
                dates=dates[name_mod],
                ref_date=ref_date,
            )
            x_dec[name_mod] += pos_encoding + date_encoding

        return self.flatten_dates(x_dec)

    def mask(
        self,
        x: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor | None], dict[str, Tensor | None]]:
        """Mask multimodal."""
        if ssl_phase == "pretrain":
            x_enc, mask_attn, mask_rec = {}, {}, {}
            for name_mod in x:
                x_enc[name_mod], mask_attn[name_mod], mask_rec[name_mod] = (
                    self.mask_mod(x[name_mod])
                )
            return x_enc, mask_attn, mask_rec

        # else, probe or finetune
        mask_attn, mask_rec = ({name_mod: None for name_mod in x},) * 2
        return x, mask_attn, mask_rec

    def unmask(
        self,
        x: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Unmask multimodal."""
        x_dec = {}
        for name_mod in x:
            mask_token = (
                self.mask_token[name_mod]
                if name_mod in self.mask_token
                else self.mask_token["shared"]
            )
            x_dec[name_mod] = self.unmask_mod(
                x[name_mod],
                mask_rec[name_mod],
                mask_token,
            )
        return x_dec

    def shuffle_mods(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Shuffle modalities and dates."""
        x = self.unflatten_dates(x)

        x_dec = self.shuffle_enc_to_dec(x)

        return self.flatten_dates(x_dec)

    def rec_pixels(
        self,
        x_dec: dict[str, Tensor],
        embed_inds: dict[str, list[Tensor]],
        mask_rec: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Reconstruct pixels from embeddings."""
        x_dec, mask_rec = self.unflatten_dates(x_dec), self.unflatten_dates(mask_rec)

        pixels_rec = {}
        for name_mod in x_dec:
            pixels_rec[name_mod] = self.embed_to_rec[name_mod](
                x_dec[name_mod],
                embed_inds[name_mod],
            )
        return pixels_rec, mask_rec

    def compute_pixels(
        self,
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        target: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Visu of reconstructed images, shared for all MIM modules."""
        pixels_masked = {}
        for name_mod in pixels_rec:
            image_size = target[name_mod].shape[3]
            out_patch_size = image_size // round(mask_rec[name_mod].shape[2] ** 0.5)
            pixels_masked[name_mod] = repeat(
                mask_rec[name_mod],
                "b d l 1 -> b d l repeat",
                repeat=out_patch_size**2,
            )  # [B L PP]
            pixels_masked[name_mod] = rearrange(
                pixels_masked[name_mod],
                "b d (h w) (p1 p2) -> b d 1 (h p1) (w p2)",
                h=image_size // out_patch_size,
                w=image_size // out_patch_size,
                p1=out_patch_size,
                p2=out_patch_size,
            )
            masked = pixels_masked[name_mod] == 1
            pixels_masked[name_mod] = torch.where(masked, 1, target[name_mod])
            pixels_rec[name_mod] = torch.where(
                masked,
                pixels_rec[name_mod],
                target[name_mod],
            )
        return pixels_masked, pixels_rec

    def compute_loss_rec(
        self,
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        target: dict[str, Tensor],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        loss_recs = []
        for name_mod in pixels_rec:
            image_size = target[name_mod].shape[3]
            out_patch_size = image_size // round(mask_rec[name_mod].shape[2] ** 0.5)
            patches_rec = rearrange(
                pixels_rec[name_mod],
                "b d c (h p1) (w p2) -> b d (h w) (p1 p2 c)",
                p1=out_patch_size,
                p2=out_patch_size,
            )
            patches_target = rearrange(
                target[name_mod],
                "b d c (h p1) (w p2) -> b d (h w) (p1 p2 c)",
                p1=out_patch_size,
                p2=out_patch_size,
            )
            if self.norm_pix_loss:
                mean = patches_target.mean(dim=-1, keepdim=True)
                var = patches_target.var(dim=-1, keepdim=True)
                patches_target = (patches_target - mean) / (var + 1.0e-6) ** 0.5

            loss_rec = self.loss_fn(patches_target - patches_rec)
            loss_rec = torch.masked_select(loss_rec, mask_rec[name_mod].bool()).mean()
            loss_recs.append(loss_rec)
        return torch.stack(loss_recs).mean()

    def compute_loss_pred(
        self,
        x_enc: dict[str, Tensor],
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        x_enc = self.unflatten_dates(x_enc)

        ref_inputs = (
            self.dataset.filter_inputs
            if self.dataset.ref_input is None
            else [self.dataset.ref_input]
        )
        x_enc = torch.cat(
            [x_enc[name_mod].flatten(1, 2) for name_mod in ref_inputs],
            dim=1,
        )

        loss_pred = 0
        for name_target, target in self.dataset.targets.items():
            targets = batch[name_target]
            logits = self.heads[name_target](x_enc, ssl_phase)

            match target.type_target:
                case "segment":
                    logits = rearrange(logits, "b 1 c h w -> (b h w) c")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.long()
                case "change_detect":
                    logits = rearrange(logits, "b 1 1 h w -> (b h w)")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.float()
                case "multilabel_classif":
                    targets = targets.float()
                case "classif":
                    targets = targets.long()

            if targets.ndim > 1:
                inds = (targets != target.missing_val).all(dim=1)
            else:
                inds = targets != target.missing_val

            inds = inds.nonzero().squeeze(dim=1)
            if len(inds) == 0:
                continue

            logits_selected = torch.index_select(
                logits,
                dim=0,
                index=inds,
            )
            targets_selected = torch.index_select(
                targets,
                dim=0,
                index=inds,
            )
            loss_pred += self.loss_pred[name_target](
                logits_selected,
                targets_selected,
            )
            self.metrics[f"{name_target}_{stage}"].update(
                logits_selected,
                targets_selected,
            )
        return loss_pred

    def encode_or_decode(
        self,
        x: dict[str, Tensor],
        model: nn.Module,
    ) -> dict[str, Tensor]:
        """Apply encoder or decoder."""
        if "multimodal" in model:
            x = self.unflatten_dates(x)
            x = self.flatten_dates(x, dim=1)

            name_mods = list(x.keys())
            split_mods = tuple(x[name_mod].shape[1] for name_mod in name_mods)

            x_mods = torch.cat([x[name_mod] for name_mod in name_mods], dim=1)
            x_mods = model["multimodal"](x_mods)
            x_mods = x_mods.split(split_mods, dim=1)
            for name_mod, x_mod in zip(name_mods, x_mods):
                x[name_mod] = x_mod

            x = self.unflatten_dates(x, dim=1)
            x = self.flatten_dates(x)
        else:
            for name_mod in x:
                model_mod = model[name_mod] if name_mod in model else model["shared"]
                x[name_mod] = model_mod(x[name_mod])
        return x

    @abstractmethod
    def mask_mod(self, x: Tensor) -> tuple[Tensor, Tensor | None, Tensor]:
        """Abstract method for modality masking."""

    @abstractmethod
    def unmask_mod(self, x: Tensor, mask_rec: Tensor, mask_token: Tensor) -> Tensor:
        """Abstract method for modality unmasking."""

    @abstractmethod
    def encode(
        self,
        x: dict[str, Tensor],
        mask_attn: dict[str, Tensor | None],
    ) -> dict[str, Tensor]:
        """Abstract method for encoding."""

    @abstractmethod
    def encoder_to_decoder(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Abstract method for encoder to decoder step."""

    @abstractmethod
    def decode(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Abstract method for decoding."""

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Shared forward pass for all MIM modules."""
        x_enc, embed_inds, dates, ref_date = self.embed(batch)
        x_enc = self.enc_add_encodings(x_enc, dates, ref_date)

        x_enc, mask_attn, mask_rec = self.mask(x_enc, ssl_phase)

        x_enc = self.encode(x_enc, mask_attn)

        if ssl_phase == "pretrain":
            x_dec = self.shuffle_mods(x_enc)
            x_dec = self.encoder_to_decoder(x_dec)
            x_dec = self.unmask(x_dec, mask_rec)
            x_dec = self.dec_add_encodings(x_dec, dates, ref_date)

            x_dec = self.decode(x_dec)

            pixels_rec, mask_rec = self.rec_pixels(x_dec, embed_inds, mask_rec)
            loss_rec = self.compute_loss_rec(
                pixels_rec,
                mask_rec,
                batch,
            )
            pixels_masked, pixels_rec = self.compute_pixels(
                pixels_rec,
                mask_rec,
                batch,
            )
            self.metrics[f"loss_rec_{stage}"].update(loss_rec)
            return loss_rec, pixels_masked, pixels_rec, None

        # else, probe or finetune
        loss_pred = self.compute_loss_pred(
            x_enc,
            batch,
            ssl_phase,
            stage,
        )
        self.metrics[f"loss_pred_{stage}"].update(loss_pred)
        return None, None, None, loss_pred
