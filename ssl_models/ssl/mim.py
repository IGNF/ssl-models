"""Generic MIM base module, inherited in specialized MIM modules."""

import itertools
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
from ssl_models import RNG
from ssl_models.layers.embed import Patchify, Pixelify
from ssl_models.layers.head import ClassificationHead, PixelifyHead
from ssl_models.layers.mask import (
    create_masked_image,
    get_cd_mask_from_logits,
    get_segment_mask_from_logits,
    get_target_mask_from_batch,
)
from ssl_models.layers.utils import (
    compute_ndvi,
    encode_dates,
    group_mods,
    max_pool,
    posemb_sincos_2d,
    reshape_encoding,
    shuffle_enc_to_dec,
    ungroup_mods,
)
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric

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
            patch_size = (
                mod.patch_size.vit
                if model in ("mae", "simmim")
                else mod.patch_size.swin
            )
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
            num_dates={
                name_mod: mod.num_dates * self.len_bands[name_mod]
                for name_mod, mod in self.dataset.inputs.items()
            },
            grid_size=self.grid_size,
        )

        # max pool mask / ndvi
        self.max_pool = {
            name_mod: partial(max_pool, patch_size=mod.patch_size.vit)
            for name_mod, mod in self.dataset.inputs.items()
        }
        self.compute_ndvi = {
            name_mod: partial(compute_ndvi, bands=mod.bands, ndvi_bands=mod.ndvi_bands)
            for name_mod, mod in self.dataset.inputs.items()
        }

        # shuffling of modalities
        self.shuffle_mods = False if model == "simmim" else multimodal == "msgfm"

        # mask tokens
        self.mask_token = nn.ParameterDict(
            {
                name_mod: nn.Parameter(
                    torch.randn(1, len_bands, 1, 1, decoder_dim),
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

        for name_loss in ("loss_rec", "loss_ndvi", "loss_pred"):
            for stage in ("train", "val", "test"):
                self.metrics[f"{name_loss}_{stage}"] = MeanMetric(
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
        x_enc, mask_token, mask_rec, embed_inds, dates = {}, {}, {}, {}, {}
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
            if f"{name_mod}_mask" in batch:
                mask_rec[name_mod] = self.max_pool[name_mod](batch[f"{name_mod}_mask"])
                mask_rec[name_mod] = mask_rec[name_mod].repeat(1, G, 1, 1)
            else:
                mask_rec[name_mod] = torch.zeros(
                    (B, D, L_out, 1),
                    dtype=torch.float,
                    device=x_enc[name_mod].device,
                )
            dates[name_mod] = batch[f"{name_mod}_dates"]

        return (
            self.group(x_enc),
            self.group(mask_token),
            self.group(mask_rec),
            embed_inds,
            dates,
            batch["ref_date"],
        )

    def enc_add_encodings(
        self,
        x_enc: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Add positional and date encodings before encoder."""
        x_enc = self.ungroup(x_enc)

        encoding = {}
        for name_mod in x_enc:
            encoding[name_mod] = reshape_encoding(
                self.enc_pos_encoding,
                self.grid_size[name_mod],
            )
            encoding[name_mod] = encoding[name_mod] + self.enc_date_encoding[name_mod](
                dates=dates[name_mod],
                ref_date=ref_date,
            )
            x_enc[name_mod] = x_enc[name_mod] + encoding[name_mod]

        return self.group(x_enc), self.group(encoding)

    def dec_add_encodings(
        self,
        x_dec: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
    ) -> dict[str, Tensor]:
        """Add positional and date encodings before decoder."""
        x_dec = self.ungroup(x_dec)

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
        mask_rec: dict[str, Tensor],
        encoding: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor | None],
        dict[str, Tensor | None],
        dict[str, Tensor | None],
        dict[str, Tensor | None],
    ]:
        """Mask multimodal."""
        if ssl_phase == "pretrain":
            mask_rec, mask_loss = self.mask_struct(x, mask_rec, stage)

            x_enc, mask_token, mask_attn = {}, {}, {}
            for name_group in x:
                (
                    x_enc[name_group],
                    mask_token[name_group],
                    mask_attn[name_group],
                    mask_rec[name_group],
                    mask_loss[name_group],
                ) = self.mask_seq(
                    x[name_group],
                    mask[name_group],
                    mask_rec[name_group],
                    mask_loss[name_group],
                    encoding[name_group],
                    name_group,
                )
            return x_enc, mask_token, mask_attn, mask_rec, mask_loss

        # else, probe or finetune
        mask_token, mask_attn, mask_rec, mask_loss = (
            {name_group: None for name_group in x},
        ) * 4
        return x, mask_token, mask_attn, mask_rec, mask_loss

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

    def shuffle_enc_to_dec(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Shuffle modalities and dates."""
        if self.shuffle_mods:
            x_dec = self.ungroup(x)

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
        mask_loss: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Reconstruct pixels from embeddings."""
        x_dec, mask_rec, mask_loss = (
            self.ungroup(x_dec),
            self.ungroup(mask_rec),
            self.ungroup(mask_loss),
        )

        pixels_rec = {}
        for name_mod in x_dec:
            (
                pixels_rec[name_mod],
                mask_rec[name_mod],
                mask_loss[name_mod],
            ) = self.embed_to_rec[name_mod](
                x_dec[name_mod],
                mask_rec[name_mod],
                mask_loss[name_mod],
                embed_inds[name_mod],
            )
        return pixels_rec, mask_rec, mask_loss

    def compute_pixels(
        self,
        pixels_rec: dict[str, Tensor],
        mask_rec,
        mask_loss: dict[str, Tensor],
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
        norm_bounds: tuple[float, float],
        idx_date_ref: int,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Visu of reconstructed images, shared for all MIM modules."""
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_mod in pixels_rec:
            if name_mod not in self.dataset.log_inputs:
                continue
            B, D, _, _, _ = batch[name_mod].shape  # noqa: N806
            inputs = torch.where(
                torch.all(mask_loss[name_mod], dim=2, keepdim=True),
                torch.nan,
                batch[name_mod],
            )
            preds = torch.where(
                torch.all(mask_loss[name_mod], dim=2, keepdim=True),
                pixels_rec[name_mod],
                batch[name_mod],
            )
            if stage == "val":
                ids = idx_date_ref
                min_b, max_b = norm_bounds[0], norm_bounds[1]
                mu, sigma = min_b, max_b - min_b
                mu = mu.view(-1, 1, 1, 1)
                sigma = sigma.view(-1, 1, 1, 1)
                batch_indices = torch.arange(B)
                img_inp = ((inputs[batch_indices, ids, :RGB_BANDS] * 5000 + 1000) - mu) / sigma
                img_pred = ((preds[batch_indices, ids, :RGB_BANDS] * 5000 + 1000) - mu) / sigma
                img_target = ((batch[name_mod][batch_indices, ids, :RGB_BANDS] * 5000 + 1000) - mu) / sigma

                img_inp = img_inp.clamp(min=min_b.view(-1,1,1,1), max=max_b.view(-1,1,1,1))
                img_pred = img_pred.clamp(min=min_b.view(-1,1,1,1), max=max_b.view(-1,1,1,1))
                img_target = img_target.clamp(min=min_b.view(-1,1,1,1), max=max_b.view(-1,1,1,1))

                log_inputs[f"{ssl_phase}_{stage}/_{name_mod}_input"] = torch.squeeze(img_inp)
                log_preds[f"{ssl_phase}_{stage}/_{name_mod}_rec"] = torch.squeeze(img_pred)
                log_targets[f"{ssl_phase}_{stage}/_{name_mod}_target"] = torch.squeeze(img_target)

            else:
                ids_ref = itertools.product(range(B), range(D))
                if mask_loss[name_mod].any():
                    ids_ref = [
                            ids for ids in ids_ref if mask_loss[name_mod][ids[0], ids[1]].any()
                        ]
                    ids = RNG.choice(ids_ref)
                    mask_ref = mask_rec[name_mod] & ~mask_loss[name_mod]
                    mask_ref = mask_ref.all(dim=2, keepdim=True)
                    mask_ref = mask_ref & ~mask_ref.all(dim=1, keepdim=True)
                    inputs_ref = torch.where(
                        mask_ref,
                        torch.nan,
                        batch[name_mod],
                    ).nanmedian(dim=1)[0][ids[0], :RGB_BANDS]
                    mu, sigma = inputs_ref.min(), inputs_ref.max() - inputs_ref.min()
                    log_inputs[f"{ssl_phase}_{stage}/_{name_mod}_input"] = (
                        inputs[ids[0], ids[1]] - mu
                    ) / sigma
                    log_preds[f"{ssl_phase}_{stage}/_{name_mod}_rec"] = (
                        preds[ids[0], ids[1]] - mu
                    ) / sigma
                    log_targets[f"{ssl_phase}_{stage}/_{name_mod}_target"] = (
                        batch[name_mod][ids[0], ids[1]] - mu
                    ) / sigma
        return log_inputs, log_preds, log_targets

    def compute_loss_rec(
        self,
        pixels_rec: dict[str, Tensor],
        mask_loss: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        losses_rec = []
        losses_ndvi = []
        weights = []
        for name_mod in pixels_rec:
            H, _ = (  # noqa: N806
                self.out_grid_size[name_mod],
                self.out_grid_size[name_mod],
            )
            P = batch[name_mod].shape[3] // H  # noqa: N806
            target = rearrange(
                batch[name_mod],
                "b d c (h p1) (w p2) -> b d (h w) (p1 p2) c",
                p1=P,
                p2=P,
            )
            if self.norm_pix_loss:
                target_groups = list(
                    torch.split(
                        target,
                        self.norm_bands[name_mod],
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

            loss_rec = self.loss_fn(target - pixels_rec[name_mod])
            loss_ndvi = self.loss_fn(
                self.compute_ndvi[name_mod](target)
                - self.compute_ndvi[name_mod](pixels_rec[name_mod]),
            )
            if mask_loss[name_mod].any():
                loss_rec = torch.masked_select(loss_rec, mask_loss[name_mod]).mean()
                loss_ndvi = torch.masked_select(loss_ndvi, mask_loss[name_mod]).mean()
                weight = mask_loss[name_mod].sum()
            else:
                loss_rec = 0 * loss_rec.mean()
                loss_ndvi = 0 * loss_ndvi.mean()
                weight = 0
            losses_rec.append(weight * loss_rec)
            losses_ndvi.append(weight * loss_ndvi)
            weights.append(weight)

        weight = sum(weights) + 1e-6
        loss_rec = torch.stack(losses_rec).sum() / weight
        loss_ndvi = torch.stack(losses_ndvi).sum() / weight
        return loss_rec, loss_ndvi, weight

    def compute_loss_pred(  # noqa: PLR0915
        self,
        x_enc: dict[str, Tensor],
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        x_enc = self.ungroup(x_enc)

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
        if not isinstance(loss_pred, Tensor):
            loss_pred = 0 * x_enc.mean()
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

    def mask_struct_random(
        self,
        x: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Structural masking of modalities, band groups, dates."""
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
            for name_mod, len_bands in self.len_bands.items():
                B, D, L, _ = mask_mod[name_mod].shape  # noqa: N806
                G = len_bands  # noqa: N806
                mask[name_mod] = torch.zeros(
                    (B, G, D // G, L),
                    dtype=torch.bool,
                )
                if self.mask_mod[name_mod]:
                    mask[name_mod] += torch.rand((B, 1, 1, 1)) < self.mask_mod[name_mod]
                if self.mask_bands[name_mod]:
                    mask[name_mod] += (
                        torch.rand((B, G, 1, 1)) < self.mask_bands[name_mod]
                    )
                if self.mask_dates[name_mod]:
                    mask[name_mod] += (
                        torch.rand((B, 1, D // G, 1)) < self.mask_dates[name_mod]
                    )
                if self.mask_loc[name_mod]:
                    mask[name_mod] += torch.rand((B, 1, 1, L)) < self.mask_loc[name_mod]
                mask[name_mod] = mask[name_mod].reshape((B, D, L, 1))

            mask = self.group(mask)

            for name_group in mask_group:
                mask_group[name_group] = torch.where(
                    mask_group[name_group].all(dim=1, keepdim=True),
                    mask[name_group],
                    mask_group[name_group],
                )

        mask_group = {
            name_group: mask_group[name_group].to(x[name_group].device)
            for name_group in mask_group
        }
        return mask_group, mask_group

    @abstractmethod
    def mask_struct(
        self,
        x: dict[str, Tensor],
        mask_rec: dict[str, Tensor],
        stage: Literal["train", "val", "test"],
    ) -> tuple[dict[str, Tensor | None], dict[str, Tensor | None]]:
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
        norm_min, norm_max, idx_date_ref = (
            batch.pop("norm_min", None),
            batch.pop("norm_max", None),
            batch.pop("idx_date_ref", None)
        )

        x_enc, mask_token, mask_rec, embed_inds, dates, ref_date = self.embed(batch)
        x_enc, encoding = self.enc_add_encodings(x_enc, dates, ref_date)

        x_enc, mask_token, mask_attn, mask_rec, mask_loss = self.mask(
            x_enc,
            mask_token,
            mask_rec,
            encoding,
            ssl_phase,
            stage,
        )

        x_enc = self.encode(x_enc, mask_attn)

        if ssl_phase == "pretrain":
            x_dec = self.shuffle_enc_to_dec(x_enc)
            x_dec = self.encoder_to_decoder(x_dec)
            x_dec = self.unmask(x_dec, mask_token, mask_rec)
            x_dec = self.dec_add_encodings(x_dec, dates, ref_date)

            x_dec = self.decode(x_dec)

            pixels_rec, mask_rec, mask_loss = self.rec_pixels(
                x_dec,
                embed_inds,
                mask_rec,
                mask_loss,
            )
            loss_rec, loss_ndvi, weight = self.compute_loss_rec(
                pixels_rec,
                mask_loss,
                batch,
            )
            log_inputs, log_preds, log_targets = self.compute_pixels(
                pixels_rec,
                mask_rec,
                mask_loss,
                batch,
                ssl_phase,
                stage,
                [norm_min, norm_max],
                idx_date_ref,
            )
            self.metrics[f"loss_rec_{stage}"].update(loss_rec, weight=weight)
            self.metrics[f"loss_ndvi_{stage}"].update(loss_ndvi, weight=weight)
            return loss_rec, log_inputs, log_preds, log_targets, None

        # else, probe or finetune
        loss_pred, log_input, log_pred, log_target = self.compute_loss_pred(
            x_enc,
            batch,
            ssl_phase,
            stage,
        )
        self.metrics[f"loss_pred_{stage}"].update(loss_pred)
        return None, log_input, log_pred, log_target, loss_pred
