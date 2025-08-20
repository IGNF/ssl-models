"""Generic MIM base module, inherited in specialized MIM modules."""

from abc import ABC, abstractmethod
from functools import partial, reduce
from math import gcd
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from torchmetrics import MeanMetric

from conf.dataset.utils import RasterConfig
from conf.datasets import DatasetsConfig
from ssl_models.layers.embed import Patchify, Pixelify
from ssl_models.layers.head import ClassificationHead, PixelifyHead
from ssl_models.layers.mask import (
    create_masked_image,
    get_cd_mask_from_logits,
    get_segment_mask_from_logits,
    get_target_mask_from_batch,
)
from ssl_models.layers.utils import (
    encode_dates,
    group_mods,
    # mask_partition,
    posemb_sincos_2d,
    reshape_encoding,
    shuffle_enc_to_dec,
    ungroup_mods,
)
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric, WeightedLoss

RGB_BANDS = 3


class BaseMIM(nn.Module, ABC):
    """Masked Auto Encoder (MAE)."""

    def __init__(  # noqa: C901, PLR0915
        self,
        datasets: DatasetsConfig,
        multimodal: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        model: Literal["mae", "mixmae"],
        num_levels: Literal[1, 3, 4],
        unpool_dim: int | None,
        embed_dim: int,
        decoder_dim: int,
        loss_type: Literal["rec", "ndvi"],
        type_head: Literal["linear", "attentive"] = "linear",
        loss_fn: Literal[torch.square, torch.abs] = torch.square,
        norm_pix_loss: bool = True,
        fac_abs_enc: float = 1.0,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
    ) -> None:
        super().__init__()
        self.dataset = datasets.dataset
        self.loss_fn = loss_fn
        self.loss_type = loss_type
        self.norm_pix_loss = norm_pix_loss
        self.stride = 2 ** (num_levels - 1)

        # patchify and pixelify
        self.num_bands = {
            name_mod: (
                [mod.bands] if isinstance(mod.bands, int) else list(map(len, mod.bands))
            )
            for name_mod, mod in datasets.dataset.inputs.items()
        }
        self.len_bands = {
            name_mod: len(num_bands) for name_mod, num_bands in self.num_bands.items()
        }
        self.norm_bands = {
            name_mod: (
                tuple(mod.norm_bands)
                if mod.norm_bands is not None
                else tuple(self.num_bands[name_mod])
            )
            for name_mod, mod in datasets.dataset.inputs.items()
        }

        self.grid_size, self.out_grid_size = {}, {}
        self.patch_embed, self.embed_to_rec = nn.ModuleDict(), nn.ModuleDict()
        for name_mod, mod in self.dataset.inputs.items():
            patch_size = mod.patch_size.mae if model == "mae" else mod.patch_size.mixmae
            self.grid_size[name_mod] = mod.image_size // patch_size
            self.out_grid_size[name_mod] = mod.image_size // (patch_size * self.stride)
            self.patch_embed[name_mod] = Patchify(
                mod.bands,
                embed_dim,
                patch_size,
                unpool_dim,
            )
            self.embed_to_rec[name_mod] = Pixelify(
                decoder_dim,
                mod.bands,
                patch_size * self.stride,
                unpool_dim,
            )

        # Fix the position encodings to sine & cosine functions
        max_grid_size = reduce(
            lambda a, b: a * b // gcd(a, b),
            self.grid_size.values(),
        )  # smallest common multiple
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
                len_bands=self.len_bands[name_mod],
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
                len_bands=self.len_bands[name_mod],
            )
            for name_mod in self.dataset.inputs
        }

        # flattening/unflattening of date dimensions
        self.group = partial(
            group_mods,
            multimodal=multimodal,
            groups=self.dataset.groups,
        )
        self.ungroup = partial(
            ungroup_mods,
            multimodal=multimodal,
            groups=self.dataset.groups,
            # num_dates={
            #     name_mod: mod.num_dates * self.len_bands[name_mod]
            #     for name_mod, mod in self.dataset.inputs.items()
            # },
            grid_size=self.grid_size,
        )
        self.group_masks = partial(
            group_mods,
            multimodal=multimodal,
            groups=self.dataset.groups_masks,
        )

        # shuffling of modalities
        self.shuffle_mods = multimodal == "msgfm"

        # mask tokens
        self.mask_token = nn.ParameterDict(
            {
                name_mod: nn.Parameter(
                    torch.randn(1, len_bands, 1, 1, embed_dim),
                )
                for name_mod, len_bands in self.len_bands.items()
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
                    type_head,
                    embed_dim * self.stride,
                    unpool_dim,
                    target.num_classes,
                    target_image_size // ref_grid_size,
                )
            else:
                self.heads[name_target] = ClassificationHead(
                    type_head,
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
                    self.loss_pred[name_target] = F.binary_cross_entropy_with_logits
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
            for stage in ("train", "val", "test"):
                self.metrics[f"{name_target}_{stage}"] = metric_partial()

        for stage in ("train", "val", "test"):
            self.metrics[f"loss_pred_{stage}"] = MeanMetric(
                dist_sync_on_step=False,
            )
            self.metrics[f"loss_rec_{stage}"] = WeightedLoss(
                dist_sync_on_step=False,
            )
            if self.loss_type == "ndvi":
                self.metrics[f"loss_ndvi_{stage}"] = WeightedLoss(
                    dist_sync_on_step=False,
                )
                self.metrics[f"loss_pretrain_{stage}"] = WeightedLoss(
                    dist_sync_on_step=False,
                )

    def embed(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, list[Tensor]],
        dict[str, Tensor],
        Tensor,
    ]:
        """Embed patches and fetch dates."""
        x_enc, embed_inds, mask_token, dates = {}, {}, {}, {}
        for name_mod in self.dataset.inputs:
            x_enc[name_mod], embed_inds[name_mod] = self.patch_embed[name_mod](
                batch[name_mod],
            )
            B = x_enc[name_mod].shape[0]  # noqa: N806
            D = x_enc[name_mod].shape[1]  # noqa: N806
            G = self.len_bands[name_mod]  # noqa: N806
            L_out = x_enc[name_mod].shape[2] // self.stride**2  # noqa: N806
            mask_token[name_mod] = self.mask_token[name_mod].to(x_enc[name_mod].dtype)
            mask_token[name_mod] = (
                mask_token[name_mod].expand((B, -1, D // G, L_out, -1)).flatten(1, 2)
            )
            dates[name_mod] = batch[f"{name_mod}_dates"]
        return (
            # self.group(x_enc),
            # self.group(mask_token),
            x_enc,
            mask_token,
            embed_inds,
            dates,
            batch["ref_date"],
        )

    def enc_add_encodings(
        self,
        x_enc: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
        num_dates: dict[str, int],
    ) -> dict[str, Tensor]:
        """Add positional and date encodings before encoder."""
        # x_enc = self.ungroup(x_enc, num_dates=num_dates)
        for name_mod, mod in x_enc.items():
            x_enc[name_mod] = rearrange(
                mod,
                "b (d l) c -> b d l c",
                d=num_dates[name_mod],
            )

        for name_mod in x_enc:
            pos_encoding = reshape_encoding(
                self.enc_pos_encoding,
                self.grid_size[name_mod],
            )
            date_encoding = self.enc_date_encoding[name_mod](
                dates=dates[name_mod],
                ref_date=ref_date,
            )
            x_enc[name_mod] = x_enc[name_mod] + pos_encoding + date_encoding

        return self.group(x_enc)

    def dec_add_encodings(
        self,
        x_dec: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
        num_dates: dict[str, int],
    ) -> dict[str, Tensor]:
        """Add positional and date encodings before decoder."""
        x_dec = self.ungroup(x_dec, num_dates=num_dates)

        for name_mod in x_dec:
            pos_encoding = reshape_encoding(
                self.dec_pos_encoding,
                self.out_grid_size[name_mod],
            )
            date_encoding = self.dec_date_encoding[name_mod](
                dates=dates[name_mod],
                ref_date=ref_date,
            )
            x_dec[name_mod] = x_dec[name_mod] + pos_encoding + date_encoding

        return self.group(x_dec)

    def mask(
        self,
        x: dict[str, Tensor],
        mask: dict[str, Tensor],
        custom_masks: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor | None],
        dict[str, Tensor | None],
        dict[str, Tensor | None],
    ]:
        """Mask multimodal."""
        if ssl_phase == "pretrain":
            x_enc, mask_token, mask_attn, mask_loss = {}, {}, {}, {}
            if not custom_masks:
                mask_rec = self.mask_struct(x)
                for name_group in x:
                    (
                        x_enc[name_group],
                        mask_token[name_group],
                        mask_attn[name_group],
                        mask_rec[name_group],
                    ) = self.mask_seq(
                        x[name_group],
                        mask[name_group],
                        mask_rec[name_group],
                        name_group,
                    )
                    mask_loss[name_group] = mask_rec[name_group]
            else:
                for name_mod, mod in custom_masks.items():
                    custom_masks[name_mod] = rearrange(mod, "b d c h w -> b d (h w) c")
                custom_masks = self.group_masks(custom_masks)
                for name_mod, mod in x.items():
                    x[name_mod] = rearrange(mod, "b d l c -> b (d l) c")
                    mask[name_mod] = rearrange(mod, "b d l c -> b (d l) c")
                mask_rec = {}
                for name_group in x:
                    (
                        x_enc[name_group],
                        mask_token[name_group],
                        mask_attn[name_group],
                        mask_rec[name_group],
                        mask_loss[name_group],
                    ) = self.custom_mask_seq(
                        x[name_group],
                        mask[name_group],
                        custom_masks.get(name_group, None),
                    )
            return x_enc, mask_token, mask_attn, mask_rec, mask_loss

        # else, probe or finetune
        mask_token, mask_attn, mask_rec, mask_rec_loss = (
            {name_group: None for name_group in x},
        ) * 4
        return x, mask_token, mask_attn, mask_rec, mask_rec_loss

    def unmask(
        self,
        x: dict[str, Tensor],
        mask: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Unmask multimodal."""
        x_dec = {}
        for name_group in x:
            x_dec[name_group] = self.unmask_seq(
                x[name_group],
                mask[name_group],
                mask_rec[name_group],
            )
        return x_dec

    def shuffle_enc_to_dec(
        self,
        x: dict[str, Tensor],
        num_dates: dict[str, int],
    ) -> dict[str, Tensor]:
        """Shuffle modalities and dates."""
        if self.shuffle_mods:
            x_dec = self.ungroup(x, num_dates=num_dates)

            x_dec = shuffle_enc_to_dec(x_dec)

            x_dec = self.group(x_dec)
        else:
            x_dec = x
        return x_dec

    def rec_pixels(
        self,
        x_dec: dict[str, Tensor],
        embed_inds: dict[str, list[Tensor]],
        mask_rec: dict[str, Tensor],
        num_dates: dict[str, int],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Reconstruct pixels from embeddings."""
        x_dec = self.ungroup(x_dec, num_dates=num_dates)
        # mask_rec = self.ungroup(mask_rec, num_dates=num_dates)
        for name_mod, mod in mask_rec.items():
            mask_rec[name_mod] = rearrange(
                mod,
                "b (d l) c -> b d l c",
                d=num_dates[name_mod],
            )
        pixels_rec = {}
        for name_mod in x_dec:
            pixels_rec[name_mod], mask_rec[name_mod] = self.embed_to_rec[name_mod](
                x_dec[name_mod],
                mask_rec[name_mod],
                embed_inds[name_mod],
            )
        return pixels_rec, mask_rec

    def compute_pixels(
        self,
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Visu of reconstructed images, shared for all MIM modules."""
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_mod in pixels_rec:
            if name_mod not in self.dataset.log_inputs:
                continue
            # inputs = torch.where(
            #     mask_rec[name_mod],
            #     0,
            #     batch[name_mod],
            # )
            inputs = torch.where(
                mask_rec[name_mod] == 2,
                0,
                batch[name_mod],
            )
            inputs[:, :, [2, 3], :, :] = torch.where(
                inputs[:, :, [2, 3], :, :] == 0,
                1,
                inputs[:, :, [2, 3], :, :],
            )
            inputs = torch.where(
                mask_rec[name_mod] == 1,
                1,
                inputs,
            )
            # inputs = torch.where(
            #     torch.all(mask_rec[name_mod], dim=2, keepdim=True),  # TODO change color
            #     1,
            #     inputs,
            # )
            preds = torch.where(
                mask_rec[name_mod] > 0,
                pixels_rec[name_mod],
                batch[name_mod],
            )
            targets = batch[name_mod]

            to_plot = mask_rec[name_mod].nonzero(as_tuple=False)
            b_to_plot = to_plot[0][0] if to_plot.numel() > 0 else 0
            d_to_plot = to_plot[0][1] if to_plot.numel() > 0 else 0

            log_inputs[f"{ssl_phase}_{stage}/_{name_mod}_input"] = inputs[
                b_to_plot,
                d_to_plot,
            ]
            log_preds[f"{ssl_phase}_{stage}/_{name_mod}_rec"] = preds[
                b_to_plot,
                d_to_plot,
            ]
            log_targets[f"{ssl_phase}_{stage}/_{name_mod}_target"] = targets[
                b_to_plot,
                d_to_plot,
            ]
        return log_inputs, log_preds, log_targets

    def compute_loss_rec(
        self,
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        mod_utilisee = "s2_zoom"
        weight = 0 if not torch.any(mask_rec[mod_utilisee].bool()) else 1
        H, _ = (  # noqa: N806
            self.out_grid_size[mod_utilisee],
            self.out_grid_size[mod_utilisee],
        )
        P = batch[mod_utilisee].shape[3] // H  # noqa: N806
        target = rearrange(
            batch[mod_utilisee],
            "b d c (h p1) (w p2) -> b d (h w) (p1 p2) c",
            p1=P,
            p2=P,
        )
        if self.norm_pix_loss:
            target_groups = list(
                torch.split(
                    target,
                    self.norm_bands[mod_utilisee],
                    dim=-1,
                ),
            )
            for idx, target_group in enumerate(target_groups):
                mean = target_group.mean(dim=(-2, -1), keepdim=True)
                var = target_group.var(dim=(-2, -1), keepdim=True)
                target_groups[idx] = (target_group - mean) / (var + 1.0e-6) ** 0.5
            target = torch.cat(target_groups, dim=-1)
        target = rearrange(
            target,
            "b d (h w) (p1 p2) c -> b d c (h p1) (w p2)",
            h=H,
            p1=P,
            p2=P,
        )
        loss_rec = self.loss_fn(target - pixels_rec[mod_utilisee])
        loss_rec = torch.masked_select(
            loss_rec,
            mask_rec[mod_utilisee],
        ).mean()
        loss_rec = torch.nan_to_num(weight * loss_rec)

        return loss_rec

    def compute_loss_ndvi(
        self,
        pixels_rec: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        mod_utilisee = "s2_zoom"
        idx_red = 2
        idx_nir = 6
        epsilon = 0.000001
        weight = 0 if not torch.any(mask_rec[mod_utilisee].bool()) else 1

        # loss ndvi
        target = batch[mod_utilisee][:, :, [idx_red, idx_nir], :, :]
        target = torch.clamp(target, min=0)
        rec = pixels_rec[mod_utilisee][:, :, [idx_red, idx_nir], :, :].float()
        rec = torch.clamp(rec, min=0)

        target_ndvi = (target[:, :, 1, :, :] - target[:, :, 0, :, :]) / (
            target[:, :, 1, :, :] + target[:, :, 0, :, :] + epsilon
        )
        rec_ndvi = (rec[:, :, 1, :, :] - rec[:, :, 0, :, :]) / (
            rec[:, :, 1, :, :] + rec[:, :, 0, :, :] + epsilon
        )

        loss_ndvi = self.loss_fn(target_ndvi - rec_ndvi)
        loss_ndvi = torch.masked_select(
            loss_ndvi,
            mask_rec[mod_utilisee][:, :, 0, :, :],
        ).mean()

        loss_ndvi = torch.nan_to_num(weight * loss_ndvi)

        # loss rec
        H, _ = (  # noqa: N806
            self.out_grid_size[mod_utilisee],
            self.out_grid_size[mod_utilisee],
        )
        P = batch[mod_utilisee].shape[3] // H  # noqa: N806
        target = rearrange(
            batch[mod_utilisee],
            "b d c (h p1) (w p2) -> b d (h w) (p1 p2) c",
            p1=P,
            p2=P,
        )
        if self.norm_pix_loss:
            target_groups = list(
                torch.split(
                    target,
                    self.norm_bands[mod_utilisee],
                    dim=-1,
                ),
            )
            for idx, target_group in enumerate(target_groups):
                mean = target_group.mean(dim=(-2, -1), keepdim=True)
                var = target_group.var(dim=(-2, -1), keepdim=True)
                target_groups[idx] = (target_group - mean) / (var + 1.0e-6) ** 0.5
            target = torch.cat(target_groups, dim=-1)
        target = rearrange(
            target,
            "b d (h w) (p1 p2) c -> b d c (h p1) (w p2)",
            h=H,
            p1=P,
            p2=P,
        )  #[:, :, [idx_red, idx_nir], :, :]
        loss_rec = self.loss_fn(target - pixels_rec[mod_utilisee])
        loss_rec = torch.masked_select(
            loss_rec,
            mask_rec[mod_utilisee],  #[:, :, [0, 1], :, :],
        ).mean()
        loss_rec = torch.nan_to_num(weight * loss_rec)

        return 0.1 * loss_ndvi + 0.9 * loss_rec, loss_ndvi, loss_rec

    def compute_loss_pred(
        self,
        x_enc: dict[str, Tensor],
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
        num_dates: dict[str, int],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        x_enc = self.ungroup(x_enc, num_dates=num_dates)

        ref_input = self.dataset.ref_input
        if ref_input is not None:
            x_ref = {}
            for name_mod in x_enc:
                D = x_enc[name_mod].shape[1]  # noqa: N806
                H = self.out_grid_size[name_mod]  # noqa: N806
                x_ref[name_mod] = rearrange(
                    x_enc[name_mod],
                    "b d (h w) c -> (b d) c h w",
                    h=H,
                )
                x_ref[name_mod] = F.interpolate(
                    x_ref[name_mod],
                    (self.out_grid_size[ref_input],) * 2,
                    mode="bilinear",
                )
                x_ref[name_mod] = rearrange(
                    x_ref[name_mod],
                    "(b d) c h w -> b d (h w) c",
                    d=D,
                )
            x_ref = torch.cat(
                [x_ref[name_mod] for name_mod in x_ref],
                dim=1,
            )

        x_enc = torch.cat(
            [x_enc[name_mod].flatten(1, 2) for name_mod in x_enc],
            dim=1,
        )
        loss_pred = 0
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_target, target in self.dataset.targets.items():
            targets = batch[name_target]
            # image logger
            if target.type_target in ("segment", "change_detect"):
                input_keys = (x for x in self.dataset.log_inputs if x in batch)
                input_img = batch[next(input_keys)][
                    0,
                    0,
                ][:RGB_BANDS]
                target_msk = get_target_mask_from_batch(
                    batch[name_target][0, 0, 0],
                    target.num_classes,
                    target.missing_val,
                )
                log_inputs[f"{ssl_phase}_{name_target}_{stage}/_input"] = input_img
                log_targets[f"{ssl_phase}_{name_target}_{stage}/_target"] = (
                    create_masked_image(
                        input_img,
                        target_msk,
                        target.num_classes,
                    )
                )
            match target.type_target:
                case "segment":
                    logits = self.heads[name_target](x_ref, ssl_phase)
                    # image_logger
                    pred_msk = get_segment_mask_from_logits(
                        logits[0, 0],
                        target.num_classes,
                    )
                    log_preds[f"{ssl_phase}_{name_target}_{stage}/_pred"] = (
                        create_masked_image(
                            input_img,
                            pred_msk,
                            target.num_classes,
                        )
                    )
                    logits = rearrange(logits, "b 1 c h w -> (b h w) c")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.long()
                case "change_detect":
                    logits = self.heads[name_target](x_ref, ssl_phase)
                    # image logger
                    input_img = batch[next(input_keys)][0, 0][:RGB_BANDS]
                    pred_msk = get_cd_mask_from_logits(logits)
                    log_preds[f"{ssl_phase}_{name_target}_{stage}/_pred"] = (
                        create_masked_image(
                            input_img,
                            pred_msk,
                            target.num_classes,
                        )
                    )
                    logits = rearrange(logits, "b 1 1 h w -> (b h w)")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.float()
                case "multilabel_classif":
                    logits = self.heads[name_target](x_enc, ssl_phase)
                    targets = targets.float()
                case "classif":
                    logits = self.heads[name_target](x_enc, ssl_phase)
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
        return loss_pred, log_inputs, log_preds, log_targets

    def encode_or_decode(
        self,
        x: dict[str, Tensor],
        model: nn.Module,
    ) -> dict[str, Tensor]:
        """Apply encoder or decoder."""
        for name_group in x:
            model_group = model[name_group] if name_group in model else model["shared"]
            x[name_group] = model_group(x[name_group])

        return x

    def encode_or_decode_all(
        self,
        x: dict[str, Tensor],
        model: nn.Module,
    ) -> dict[str, Tensor]:
        """Apply encoder or decoder on all modalities."""
        name_groups = list(x.keys())
        split_groups = tuple(x[name_group].shape[1] for name_group in name_groups)

        x_groups = torch.cat([x[name_group] for name_group in name_groups], dim=1)
        x_groups = model(x_groups)
        x_groups = x_groups.split(split_groups, dim=1)
        for name_group, x_group in zip(name_groups, x_groups):
            x[name_group] = x_group

        return x

    @abstractmethod
    def mask_struct(self, x: Tensor) -> dict[str, Tensor | None]:
        """Abstract method for structural masking."""

    @abstractmethod
    def mask_seq(
        self,
        x: Tensor,
        mask_token: Tensor,
        mask_rec: Tensor | None,
        name_group: str,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        """Abstract method for masking of sequence."""

    @abstractmethod
    def custom_mask_seq(
        self,
        x: Tensor,
        mask_token: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        """Abstract method for masking of sequence."""

    @abstractmethod
    def unmask_seq(self, x: Tensor, mask_token: Tensor, mask_rec: Tensor) -> Tensor:
        """Abstract method for unmasking of sequence."""

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
        num_dates = {
            name_mod: batch[name_mod].shape[1] * self.len_bands[name_mod]
            for name_mod, mod in self.dataset.inputs.items()
        }
        masks = {k: batch.pop(k) for k in self.dataset.masks}
        masks_test_inference = {
            k: batch.pop(k) for k, v in batch.copy().items() if k.endswith("test")
        }

        x_enc, mask_token, embed_inds, dates, ref_date = self.embed(batch)
        x_enc, mask_token, mask_attn, mask_rec, mask_loss = self.mask(
            x_enc,
            mask_token,
            masks,
            ssl_phase,
        )
        x_enc = self.enc_add_encodings(x_enc, dates, ref_date, num_dates)
        x_enc = self.encode(x_enc, mask_attn)
        if ssl_phase == "pretrain":
            x_dec = self.shuffle_enc_to_dec(x_enc, num_dates)
            x_dec = self.encoder_to_decoder(x_dec)
            x_dec = self.dec_add_encodings(x_dec, dates, ref_date, num_dates)
            x_dec = self.decode(x_dec)
            pixels_rec, mask_loss = self.rec_pixels(
                x_dec,
                embed_inds,
                mask_loss,
                num_dates,
            )
            if stage == "test":
                mask_loss.update(
                    {
                        k.replace("_mask_test", ""): v
                        for k, v in masks_test_inference.items()
                    },
                )

            nb_pixels_loss = mask_loss["s2_zoom"][:, :, 0, :, :].sum()  # TODO
            if self.loss_type == "ndvi":
                loss, loss_ndvi, loss_rec = self.compute_loss_ndvi(
                    pixels_rec,
                    mask_loss,
                    batch,
                )
                self.metrics[f"loss_pretrain_{stage}"].update(loss, nb_pixels_loss)
                self.metrics[f"loss_rec_{stage}"].update(loss_rec, nb_pixels_loss)
                self.metrics[f"loss_ndvi_{stage}"].update(loss_ndvi, nb_pixels_loss)
            else:
                loss = self.compute_loss_rec(
                    pixels_rec,
                    mask_loss,
                    batch,
                )
                self.metrics[f"loss_rec_{stage}"].update(loss, nb_pixels_loss)

            _, mask_rec = self.rec_pixels(x_dec, embed_inds, mask_rec, num_dates)
            mask_log = {
                "s2_zoom": mask_rec["s2_zoom"].long() + mask_loss["s2_zoom"].long()
            }
            log_inputs, log_preds, log_targets = self.compute_pixels(
                pixels_rec,
                mask_log,
                batch,
                ssl_phase,
                stage,
            )

            return loss, log_inputs, log_preds, log_targets, None

        # else, probe or finetune
        loss_pred, log_input, log_pred, log_target = self.compute_loss_pred(
            x_enc,
            batch,
            ssl_phase,
            stage,
            num_dates,
        )
        self.metrics[f"loss_pred_{stage}"].update(loss_pred)
        return None, log_input, log_pred, log_target, loss_pred
