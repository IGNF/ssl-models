"""Generic MIM base module, inherited in specialized MIM modules."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat
from torch import Tensor, nn

from conf.dataset.utils import RasterConfig
from conf.datasets import DatasetsConfig
from ssl_models.layers.embed import Patchify, Pixelify
from ssl_models.layers.head import AttentiveHead, LinearHead
from ssl_models.layers.utils import posemb_sincos_2d
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric


class BaseMIM(nn.Module, ABC):
    """Masked Auto Encoder (MAE)."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        in_chans: int,
        image_size: int,
        patch_size: int,
        out_patch_size: int,
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

        target_patch_size = datasets.dataset.target_patch_size
        head_dim = embed_dim * 2 ** (num_levels - 1)

        self.dataset = datasets.dataset
        self.heads = nn.ModuleDict()
        for name_target, target in self.dataset.targets.items():
            if isinstance(target, RasterConfig):
                self.heads[name_target] = Pixelify(
                    head_dim,
                    target.num_classes,
                    target_patch_size,
                    unpool=False,
                )
            elif type_head == "attentive":
                self.heads[name_target] = AttentiveHead(
                    head_dim,
                    target.num_classes,
                )
            else:
                self.heads[name_target] = LinearHead(
                    head_dim,
                    target.num_classes,
                )

        self.sigmoid_pred = {}
        self.loss_pred = {}
        self.metrics = nn.ModuleDict()
        for name_target, target in datasets.dataset.targets.items():
            match target.type_target:
                case "classif" | "segment":
                    self.sigmoid_pred[name_target] = lambda x: x
                    self.loss_pred[name_target] = F.cross_entropy
                    metric_partial = partial(
                        MonoLabelMetric,
                        type_target=target.type_target,
                        num_classes=target.num_classes,
                    )
                case "change_detect":
                    self.sigmoid_pred[name_target] = torch.sigmoid
                    self.loss_pred[name_target] = F.binary_cross_entropy
                    metric_partial = partial(
                        MonoLabelMetric,
                        type_target=target.type_target,
                        num_classes=target.num_classes,
                    )
                case "multilabel_classif":
                    self.sigmoid_pred[name_target] = torch.sigmoid
                    self.loss_pred[name_target] = F.binary_cross_entropy
                    metric_partial = partial(
                        MultiLabelMetric,
                        num_classes=target.num_classes,
                    )
            for stage in ("train", "val"):
                self.metrics[f"{name_target}_{stage}"] = metric_partial()

    def compute_pixels(
        self,
        patches_rec: Tensor,
        mask_rec: Tensor,
        target: Tensor,
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

    def compute_loss_rec(
        self,
        patches_rec: Tensor,
        mask_rec: Tensor,
        target: Tensor,
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
            logits = self.sigmoid_pred[name_target](logits)
            targets = batch[name_target]
            match target.type_target:
                case "segment":
                    # logits = rearrange(logits, "b h (w c) -> (b h w) c", c=target.num_classes, w=target.image_size, h=target.image_size)
                    logits = rearrange(
                        logits, "b (w h) (p1 p2 c) -> (b h p1 w p2) c",
                        c=target.num_classes,
                        p1=self.out_patch_size,
                        p2=self.out_patch_size,
                        w=target.image_size // self.out_patch_size
                    )
                    targets = rearrange(targets, "b c h w -> (b c h w)")
                    targets = targets.long()
                case "change_detect":
                    logits = rearrange(logits, "b 1 h w -> (b h w)")
                    targets = rearrange(targets, "b h w -> (b h w)")
                    targets = targets.float()
                case "multilabel_classif":
                    targets = targets.float()
                case _:
                    pass

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

        return loss_pred.mean()

    def forward(
        self,
        batch: dict[str, Tensor],
        mod_name: str,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Shared forward pass for all MIM modules."""
        x_enc, embed_inds = self.patch_embed(batch[mod_name])
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

            loss_rec = self.compute_loss_rec(
                patches_rec,
                mask_rec,
                batch[mod_name],
            )
            pixels_target, pixels_rec = self.compute_pixels(
                patches_rec,
                mask_rec,
                batch[mod_name],
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
