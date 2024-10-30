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
from ssl_models.layers.head import AttentiveHead, LinearHead
from ssl_models.layers.utils import posemb_sincos_2d
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric


class BaseMIM(nn.Module, ABC):
    """Masked Auto Encoder (MAE)."""

    def __init__(  # noqa: C901
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
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.norm_pix_loss = norm_pix_loss
        self.stride = 2 ** (num_levels - 1)

        self.dataset = datasets.dataset
        self.num_bands = {}
        self.patch_size, self.out_patch_size, self.grid_size, self.image_size = (
            {},
            {},
            {},
            {},
        )
        for name_mod, mod in self.dataset.inputs.items():
            self.num_bands[name_mod] = (
                mod.bands if isinstance(mod.bands, int) else len(mod.bands)
            )
            self.image_size[name_mod] = mod.image_size
            self.patch_size[name_mod] = (
                mod.patch_size.mae if model == "mae" else mod.patch_size.mixmae
            )
            self.out_patch_size[name_mod] = self.patch_size[name_mod] * self.stride
            self.grid_size[name_mod] = mod.image_size // self.patch_size[name_mod]

        self.max_grid_size = max(self.grid_size.values())
        self.grid_fac = {
            name_mod: self.max_grid_size
            // (mod.image_size // self.patch_size[name_mod])
            for name_mod, mod in self.dataset.inputs.items()
        }

        # patchify and pixelify
        self.patch_embed, self.embed_to_rec = nn.ModuleDict(), nn.ModuleDict()
        for name_mod in self.dataset.inputs:
            self.patch_embed[name_mod] = Patchify(
                self.num_bands[name_mod],
                embed_dim,
                self.patch_size[name_mod],
                unpool,
            )
            self.embed_to_rec[name_mod] = Pixelify(
                decoder_dim,
                self.num_bands[name_mod],
                self.patch_size[name_mod] * self.stride,
                unpool,
            )

        # Fix the position encodings to sine & cosine functions
        self.register_buffer(
            name="enc_pos_encoding",
            tensor=posemb_sincos_2d(
                h=self.max_grid_size,
                w=self.max_grid_size,
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
                h=self.max_grid_size,
                w=self.max_grid_size,
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

        self.dataset = datasets.dataset
        self.heads = nn.ModuleDict()
        for name_target, target in self.dataset.targets.items():
            if isinstance(target, RasterConfig):
                self.heads[name_target] = Pixelify(
                    embed_dim * self.stride,
                    target.num_classes,
                    None,
                    unpool=False,
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

    def compute_pixels(
        self,
        patches_rec: Tensor,
        mask_rec: Tensor,
        target: Tensor,
        out_patch_size: int,
        image_size: int,
    ) -> Tensor:
        """Shared visu of reconstruction images shown for all MIM modules."""
        mask_pixels = repeat(
            mask_rec,
            "b l 1 -> b l repeat",
            repeat=out_patch_size**2,
        )  # [B L PP]
        mask_pixels = rearrange(
            mask_pixels,
            "b (h w) (p1 p2) -> b 1 (h p1) (w p2)",
            h=image_size // out_patch_size,
            w=image_size // out_patch_size,
            p1=out_patch_size,
            p2=out_patch_size,
        )
        pixels_target = torch.where(mask_pixels == 1, 1, target)

        pixels_rec = rearrange(
            patches_rec,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=image_size // out_patch_size,
            w=image_size // out_patch_size,
            p1=out_patch_size,
            p2=out_patch_size,
        )
        pixels_rec = torch.where(mask_pixels == 1, pixels_rec, target)

        return pixels_target, pixels_rec

    def compute_loss_rec(
        self,
        patches_rec: Tensor,
        mask_rec: Tensor,
        target: Tensor,
        out_patch_size: int,
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        patches_target = rearrange(
            target,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=out_patch_size,
            p2=out_patch_size,
        )

        if self.norm_pix_loss:
            mean = patches_target.mean(dim=-1, keepdim=True)
            var = patches_target.var(dim=-1, keepdim=True)
            patches_target = (patches_target - mean) / (var + 1.0e-6) ** 0.5

        loss_rec = self.loss_fn(patches_target - patches_rec)
        loss_rec = torch.masked_select(loss_rec, mask_rec.bool())
        loss_rec = loss_rec.mean()

        self.metrics[f"loss_rec_{stage}"].update(loss_rec)
        return loss_rec

    def compute_loss_pred(
        self,
        x_enc: Tensor,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        loss_pred = 0
        for name_target, target in self.dataset.targets.items():
            logits = self.heads[name_target](x_enc, ssl_phase)
            targets = batch[name_target]
            target_patch_size = int(
                target.image_size / self.image_size * self.out_patch_size,
            )
            match target.type_target:
                case "segment":
                    logits = rearrange(
                        logits,
                        "b (h w) (p1 p2 c) -> (b h p1 w p2) c",
                        c=target.num_classes,
                        p1=target_patch_size,
                        p2=target_patch_size,
                        w=self.image_size // target_patch_size,
                    )
                    targets = rearrange(targets, "b c h w -> (b c h w)")
                    targets = targets.long()
                case "change_detect":
                    logits = rearrange(logits, "b 1 h w -> (b h w)")
                    targets = rearrange(targets, "b h w -> (b h w)")
                    targets = targets.float()
                case "multilabel_classif":
                    targets = targets.float()

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

        self.metrics[f"loss_pred_{stage}"].update(loss_pred)
        return loss_pred

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Shared forward pass for all MIM modules."""
        x_enc, embed_inds = {}, {}
        for name_mod, mod in self.dataset.inputs.items():
            for idx in range(mod.num_dates):
                x_enc[(name_mod, idx)], embed_inds[(name_mod, idx)] = self.patch_embed[
                    name_mod
                ](
                    batch[name_mod][:, idx],
                )
                pos_encoding = self.enc_pos_encoding
                pos_encoding = pos_encoding.unflatten(2, (-1, self.grid_fac[name_mod]))
                pos_encoding = pos_encoding.unflatten(1, (-1, self.grid_fac[name_mod]))
                pos_encoding = pos_encoding[:, :, 0, :, 0, :].flatten(1, 2)
                x_enc[(name_mod, idx)] = x_enc[(name_mod, idx)] + pos_encoding

        if ssl_phase == "pretrain":
            x_enc, mask_s1, mask_s2, mask_s3, mask_s4, mask_rec = self.mask_or_mix(
                x_enc,
            )
        else:
            mask_s1, mask_s2, mask_s3, mask_s4 = (None,) * 4

        x_enc = self.encode(
            x_enc,
            mask_s1=mask_s1,
            mask_s2=mask_s2,
            mask_s3=mask_s3,
            mask_s4=mask_s4,
            ssl_phase=ssl_phase,
        )

        pixels_target, pixels_rec = {}, {}
        if ssl_phase == "pretrain":
            loss_rec = 0
            for name_mod, mod in self.dataset.inputs.items():
                for idx in range(mod.num_dates):
                    x_dec = self.encoder_to_decoder(x_enc[(name_mod, idx)])
                    x_dec = self.unmask_or_unmix(x_dec, mask_rec[(name_mod, idx)])

                    pos_encoding = self.dec_pos_encoding
                    pos_encoding = pos_encoding.unflatten(
                        2,
                        (-1, self.grid_fac[name_mod]),
                    )
                    pos_encoding = pos_encoding.unflatten(
                        1,
                        (-1, self.grid_fac[name_mod]),
                    )
                    pos_encoding = pos_encoding[:, :, 0, :, 0, :].flatten(1, 2)
                    x_dec = x_dec + pos_encoding

                    x_dec = self.decode(x_dec)

                    patches_rec = self.embed_to_rec[name_mod](
                        x_dec,
                        embed_inds[(name_mod, idx)],
                    )

                    loss_rec += self.compute_loss_rec(
                        patches_rec,
                        mask_rec[(name_mod, idx)],
                        batch[name_mod][:, idx],
                        out_patch_size=self.out_patch_size[name_mod],
                        stage=stage,
                    )
                    pixels_target[(name_mod, idx)], pixels_rec[(name_mod, idx)] = (
                        self.compute_pixels(
                            patches_rec,
                            mask_rec[(name_mod, idx)],
                            batch[name_mod][:, idx],
                            out_patch_size=self.out_patch_size[name_mod],
                            image_size=self.image_size[name_mod],
                        )
                    )
            return loss_rec, pixels_target, pixels_rec, None

        loss_pred = self.compute_loss_pred(
            x_enc,
            batch,
            ssl_phase=ssl_phase,
            stage=stage,
        )
        return None, None, None, loss_pred

    @abstractmethod
    def mask_or_mix(self, x: Tensor):  # noqa: ANN201
        """Abstract method for masking/mixing."""

    @abstractmethod
    def unmask_or_unmix(self, x: Tensor, mask_rec: Tensor) -> Tensor:
        """Abstract method for unmasking/unmixing."""

    @abstractmethod
    def encode(self, x: Tensor, *args) -> Tensor:  # noqa: ANN002
        """Abstract method for encoding."""

    @abstractmethod
    def encoder_to_decoder(self, x: Tensor) -> Tensor:
        """Abstract method for encoder to decoder step."""

    @abstractmethod
    def decode(self, x: Tensor) -> Tensor:
        """Abstract method for decoding."""
