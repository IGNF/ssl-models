"""Base module for foundation model baselines."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor
from torch.nn import Module, ModuleDict
from torchmetrics import MeanMetric

from conf.dataset.utils import RasterConfig
from conf.datasets import DatasetsConfig
from ssl_models.layers.head import ClassificationHead, PixelifyHead
from ssl_models.layers.mask import (
    create_masked_image,
    get_cd_mask_from_logits,
    get_segment_mask_from_logits,
    get_target_mask_from_batch,
)
from ssl_models.layers.utils import encode_dates, group_mods, ungroup_mods
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric

RGB_BANDS = 3


class BaseModule(Module, ABC):
    """Base Module for baselines."""

    def __init__(  # noqa: C901
        self,
        datasets: DatasetsConfig,
        patch_size: int,
        embed_dim: int,
        unpool_dim: int,
        type_head: Literal["linear", "attentive"] = "linear",
        multimodal: Literal["shared", "monotemp", "croma-intergroup"] = "shared",
        add_date_enc: bool = True,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(**kwargs)

        self.dataset = datasets.dataset
        self.type_head = type_head
        self.patch_size = patch_size
        self.multimodal = multimodal
        self.keep_norm = keep_norm
        self.add_date_enc = add_date_enc

        self.num_bands = {
            name_mod: (
                [mod.bands] if isinstance(mod.bands, int) else list(map(len, mod.bands))
            )
            for name_mod, mod in datasets.dataset.inputs.items()
        }

        self.len_bands = {
            name_mod: len(num_bands) for name_mod, num_bands in self.num_bands.items()
        }
        self.grid_size = {}

        for name_mod, mod in self.dataset.inputs.items():
            self.grid_size[name_mod] = mod.image_size // self.patch_size

        self.num_dates = {
            name_mod: mod.num_dates * self.len_bands[name_mod]
            for name_mod, mod in self.dataset.inputs.items()
        }
        # heads
        self.heads = ModuleDict()
        for name_target, target in self.dataset.targets.items():
            if isinstance(target, RasterConfig):
                if self.dataset.ref_input is None:
                    msg = f"Ref input must be provided for raster target {name_target}"
                    raise ValueError(msg)
                target_image_size = round(
                    self.dataset.crop_meters / target.resolution_meters,
                )
                ref_grid_size = self.grid_size[self.dataset.ref_input]
                if target_image_size % ref_grid_size:
                    msg = (
                        f"Target image size {target_image_size} "
                        f"is not a multiple of ref input grid {ref_grid_size}"
                    )
                    raise ValueError(msg)
                self.heads[name_target] = PixelifyHead(
                    type_head,
                    embed_dim,
                    unpool_dim,
                    target.num_classes,
                    target_image_size // ref_grid_size,
                )
            else:
                self.heads[name_target] = ClassificationHead(
                    type_head,
                    embed_dim,
                    target.num_classes,
                )

        self.loss_pred = {}
        self.metrics = ModuleDict()
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

        for name_loss in ("loss_rec", "loss_pred"):
            for stage in ("train", "val", "test"):
                self.metrics[f"{name_loss}_{stage}"] = MeanMetric(
                    dist_sync_on_step=False,
                )

        # flattening/unflattening of date dimensions
        self.group = partial(
            group_mods,
            multimodal=self.multimodal,
            groups=self.dataset.groups,
        )
        self.ungroup = partial(
            ungroup_mods,
            multimodal=self.multimodal,
            groups=self.dataset.groups,
            num_dates=self.num_dates,
            grid_size=self.grid_size,
        )

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

    @abstractmethod
    def _build_backbone(self) -> None:
        return

    @abstractmethod
    def freeze_backbone(self) -> None:
        return

    @abstractmethod
    def grouped_parameters(self, base_lr: float, rate_decay: float = 0.75) -> None:
        return

    def _add_date_encodings(
        self,
        x_enc: dict[str, Tensor],
        dates: dict[str, Tensor],
        ref_date: Tensor,
    ) -> dict[str, Tensor]:
        """Add positional and date encodings before encoder."""
        x_enc = self.ungroup(x_enc)
        dates_k = set(dates.keys())
        x_k = set(x_enc.keys())

        for name_mod in dates_k.intersection(x_k):
            date_encoding = self.enc_date_encoding[name_mod](
                dates=dates[name_mod],
                ref_date=ref_date,
            )
            x_enc[name_mod] = x_enc[name_mod] + date_encoding

        return self.group(x_enc)

    def compute_logits(
        self,
        x_enc: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> dict[str, Tensor]:
        x_enc = self.ungroup(x_enc)
        ref_input = self.dataset.ref_input
        if ref_input is not None:
            x_ref = {}
            for name_mod in x_enc:
                D = x_enc[name_mod].shape[1]  # noqa: N806
                H = self.grid_size[name_mod]  # noqa: N806
                x_ref[name_mod] = rearrange(
                    x_enc[name_mod],
                    "b d (h w) c -> (b d) c h w",
                    h=H,
                )
                x_ref[name_mod] = F.interpolate(
                    x_ref[name_mod],
                    (self.grid_size[ref_input],) * 2,
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
        logits = {}
        for name_target, target in self.dataset.targets.items():
            match target.type_target:
                case "segment" | "change_detect":
                    logits[name_target] = self.heads[name_target](x_ref, ssl_phase)
                case "multilabel_classif" | "classif":
                    logits[name_target] = self.heads[name_target](x_enc, ssl_phase)
        return logits

    def compute_loss_pred(
        self,
        x_enc: dict[str, Tensor],
        batch: dict[str, Tensor],
        logits: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[
        Tensor,
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
    ]:
        """Shared loss computation for all MIM modules."""
        logits = self.compute_logits(x_enc, ssl_phase)

        loss_pred = 0
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_target, target in self.dataset.targets.items():
            logits_target = logits[name_target]
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
                    # image_logger
                    pred_msk = get_segment_mask_from_logits(
                        logits_target[0, 0],
                        target.num_classes,
                    )
                    log_preds[f"{ssl_phase}_{name_target}_{stage}/_pred"] = (
                        create_masked_image(
                            input_img,
                            pred_msk,
                            target.num_classes,
                        )
                    )
                    logits_target = rearrange(logits_target, "b 1 c h w -> (b h w) c")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.long()
                case "change_detect":
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
                    logits_target = rearrange(logits_target, "b 1 1 h w -> (b h w)")
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
                logits_target,
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
            loss_pred = 0 * list(logits.values()).pop().mean()
        self.metrics[f"loss_pred_{stage}"].update(loss_pred)

        return loss_pred, logits, log_inputs, log_preds, log_targets

    def resize_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Resize input rasters based on dataset config."""
        for name_mod, mod in self.dataset.inputs.items():
            batch[name_mod] = F.interpolate(
                batch[name_mod].flatten(0, 1),
                size=(mod.image_size,) * 2,
                mode="nearest",
            ).unflatten(0, (-1, mod.num_dates))
        return batch
