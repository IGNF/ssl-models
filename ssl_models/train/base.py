"""Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model."""

from abc import ABC
from functools import partial
from math import isqrt
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric

from ssl_models.conf.dataset.utils import DatasetConfig, RasterConfig
from ssl_models.conf.datasets import DatasetsConfig
from ssl_models.layers.overlay import (
    create_overlay,
    onehot_cd_from_logits,
    onehot_pred_from_logits,
    onehot_target_from_batch,
)
from ssl_models.train.feature_utils import batch_pca_reduce
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric

RGB_BANDS = 3


class BaseModule(LightningModule, ABC):
    """Lightning base module."""

    def __init__(self, datasets: DatasetsConfig) -> None:
        super().__init__()
        self.datasets = datasets

        self.metrics = nn.ModuleDict()
        self.loss_pred = {}

        for name_dataset, dataset in datasets.datasets.items():
            self._init_dataset(name_dataset, dataset)

    def _init_dataset(
        self,
        name_dataset: str,
        dataset: DatasetConfig,
    ) -> None:
        for name_target, target in dataset.targets.items():
            if name_target in self.loss_pred:
                msg = "Target is defined multiple times"
                raise ValueError(msg)

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
                self.metrics[f"{name_loss}_{name_dataset}_{stage}"] = MeanMetric(
                    dist_sync_on_step=False,
                )

    @staticmethod
    def _to_cpu_tensors(log_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Detach and move logging tensors to CPU to lower peak GPU memory."""
        return {name: tensor.detach().cpu() for name, tensor in log_dict.items()}

    def compute_logs_pred(
        self,
        batch: dict[str, Tensor],
        logits: dict[str, Tensor],
        name_dataset: str,
        dataset: DatasetConfig,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
    ]:
        """Visu of predicted rasters."""
        log_inputs, log_preds, log_targets = {}, {}, {}
        for name_target, target in dataset.targets.items():
            if target.type_target in ("segment", "change_detect"):
                overlay_img = batch[dataset.log_inputs[0]][
                    0,
                    0,
                    :RGB_BANDS,
                ]
                onehot_target = onehot_target_from_batch(
                    batch[name_target][0, 0, 0],
                    target.num_classes,
                    target.missing_val,
                )
                log_inputs[
                    f"{ssl_phase}_{name_dataset}_{name_target}_{stage}/_input"
                ] = overlay_img
                log_targets[
                    f"{ssl_phase}_{name_dataset}_{name_target}_{stage}/_target"
                ] = create_overlay(
                    overlay_img,
                    onehot_target,
                    target.num_classes,
                )
            match target.type_target:
                case "segment":
                    onehot_pred = onehot_pred_from_logits(
                        logits[name_target][0, 0],
                        target.num_classes,
                    )
                    log_preds[
                        f"{ssl_phase}_{name_dataset}_{name_target}_{stage}/_pred"
                    ] = create_overlay(
                        overlay_img,
                        onehot_pred,
                        target.num_classes,
                    )
                case "change_detect":
                    overlay_img = batch[dataset.log_inputs[1]][
                        0,
                        0,
                        :RGB_BANDS,
                    ]
                    onehot_pred = onehot_cd_from_logits(logits)
                    log_preds[
                        f"{ssl_phase}_{name_dataset}_{name_target}_{stage}/_pred"
                    ] = create_overlay(
                        overlay_img,
                        onehot_pred,
                        target.num_classes,
                    )
        return log_inputs, log_preds, log_targets

    def compute_loss_pred(
        self,
        batch: dict[str, Tensor],
        logits: dict[str, Tensor],
        name_dataset: str,
        dataset: DatasetConfig,
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Compute prediction loss."""
        loss_pred = 0
        weight = 0
        for name_target, target in dataset.targets.items():
            logits_target = logits[name_target]
            targets = batch[name_target]

            match target.type_target:
                case "segment":
                    logits_target = rearrange(logits_target, "b 1 c h w -> (b h w) c")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.long()
                case "change_detect":
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

            weight += batch[name_target].shape[0] * inds.float().mean()

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

        self.metrics[f"loss_pred_{name_dataset}_{stage}"].update(loss_pred)
        return loss_pred, weight

    def probe_or_finetune_step(
        self,
        batch: dict[str, dict[str, Tensor]],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Probing/finetuning step."""
        if (
            self.trainer.ssl_phase == "finetune"
            and stage != "train"
            and self.ema_model is not None
        ):
            model = self.ema_model
        else:
            model = self.model

        losses_pred, weights = {}, {}
        logs_inputs, logs_preds, logs_targets = {}, {}, {}
        feature_maps = {}  # Store PCA-reduced feature maps for logging

        for name_dataset, dataset in self.datasets.datasets.items():
            if batch[name_dataset] is None:
                continue

            # Extract features during test phase
            rgb_features: dict[str, Tensor] = {}
            if stage == "test":
                batch[name_dataset], _, _, feat_dict, logits = model(
                    batch[name_dataset],
                    name_dataset,
                    ssl_phase=self.trainer.ssl_phase,
                )
                if feat_dict and any(v is not None for v in feat_dict.values()):
                    rgb_features = self._features_to_rgb(
                        feat_dict,
                        batch_modalities=batch[name_dataset],
                    )
            else:
                batch[name_dataset], _, _, _, logits = model(
                    batch[name_dataset],
                    name_dataset,
                    ssl_phase=self.trainer.ssl_phase,
                )

            loss_pred, weight = self.compute_loss_pred(
                batch[name_dataset],
                logits,
                name_dataset,
                dataset,
                stage=stage,
            )
            losses_pred[name_dataset] = loss_pred
            weights[name_dataset] = weight

            log_inputs, log_preds, log_targets = self.compute_logs_pred(
                batch[name_dataset],
                logits,
                name_dataset,
                dataset,
                ssl_phase=self.trainer.ssl_phase,
                stage=stage,
            )
            # Keep visualization artifacts on CPU as soon as possible.
            log_inputs = self._to_cpu_tensors(log_inputs)
            log_preds = self._to_cpu_tensors(log_preds)
            log_targets = self._to_cpu_tensors(log_targets)
            logs_inputs.update(log_inputs)
            logs_preds.update(log_preds)
            logs_targets.update(log_targets)

            # Group feature maps with matching input/target under the same prefix
            if stage == "test":
                ssl_phase = self.trainer.ssl_phase
                for name_target, target in dataset.targets.items():
                    if not isinstance(target, RasterConfig):
                        continue
                    prefix = f"{ssl_phase}_{name_dataset}_{name_target}_test"
                    input_key = f"{prefix}/_input"
                    target_key = f"{prefix}/_target"
                    if input_key in log_inputs:
                        feature_maps[input_key] = log_inputs[input_key]
                    for mod_name, rgb in rgb_features.items():
                        feature_maps[f"{prefix}/_feature_{mod_name}"] = rgb
                    if target_key in log_targets:
                        feature_maps[target_key] = log_targets[target_key]

        loss_pred = sum(
            losses_pred[name_dataset] * weights[name_dataset]
            for name_dataset in losses_pred
        ) / sum(weights.values())
        return {
            "loss": loss_pred,
            "losses": losses_pred,
            "log_inputs": logs_inputs,
            "log_preds": logs_preds,
            "log_targets": logs_targets,
            "feature_maps": feature_maps,
        }

    @staticmethod
    def _get_spatial_dims_from_batch_modality(
        modality: str,
        batch_modalities: dict[str, Tensor] | None,
    ) -> tuple[int, int] | None:
        min_spatial_ndim = 4
        if batch_modalities is None or modality not in batch_modalities:
            return None

        modality_tensor = batch_modalities[modality]
        if (
            not isinstance(modality_tensor, Tensor)
            or modality_tensor.ndim < min_spatial_ndim
        ):
            return None

        return modality_tensor.shape[-2], modality_tensor.shape[-1]

    def _features_to_rgb(
        self,
        feat_dict: dict[str, Tensor],
        batch_modalities: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Convert feature maps to RGB using PCA.

        Parameters
        ----------
        feat_dict : dict[str, Tensor]
            Dictionary of feature tensors, typically keyed by modality name.
            Each tensor should have shape (B, L, D) or (B, C, H, W).
        batch_modalities : dict[str, Tensor], optional
            Dataset batch dictionary used to recover target modality image size.

        Returns
        -------
        dict[str, Tensor]
            Dictionary of RGB feature visualizations (B, 3, H, W) format.
            Only the first sample in the batch is kept to reduce memory.
        """
        rgb_dict = {}
        dim_vec, dim_tokens, dim_spatial = 2, 3, 4
        for modality, feature_tensor in feat_dict.items():
            if feature_tensor is None or feature_tensor.numel() == 0:
                continue

            try:
                # Only keep first sample
                features = feature_tensor[:1]
                spatial_dims_mod: tuple[int, int] | None = None

                # Ensure features are 3D (B, L, D) or can be reshaped to it
                if features.dim() == dim_vec:
                    # (B, D) -> (B, 1, D)
                    features = features.unsqueeze(1)
                elif features.dim() == dim_spatial:
                    # (B, C, H, W) -> (B, H*W, C)
                    B, C, H, W = features.shape
                    spatial_dims_mod = (H, W)
                    features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
                elif features.dim() == dim_tokens:
                    # If tokens form a square grid (e.g. 256 -> 16x16),
                    # keep spatial layout.
                    num_tokens = features.shape[1]
                    side = isqrt(num_tokens)
                    if side * side == num_tokens:
                        spatial_dims_mod = (side, side)
                elif features.dim() != dim_tokens:
                    # Unexpected shape, skip
                    continue

                # Apply PCA reduction
                rgb_features = batch_pca_reduce(
                    {modality: features},
                    spatial_dims={modality: spatial_dims_mod}
                    if spatial_dims_mod is not None
                    else None,
                )

                target_dims = self._get_spatial_dims_from_batch_modality(
                    modality,
                    batch_modalities,
                )
                if target_dims is not None and modality in rgb_features:
                    rgb_map = rgb_features[modality]
                    if (
                        rgb_map.dim() == dim_spatial
                        and rgb_map.shape[-2:] != target_dims
                    ):
                        rgb_features[modality] = F.interpolate(
                            rgb_map,
                            size=target_dims,
                            mode="bilinear",
                            align_corners=False,
                        )

                rgb_dict.update(rgb_features)
            except Exception as e:
                # Log warning but don't crash
                print(f"Warning: Failed to process features for {modality}: {e}")
                continue

        return rgb_dict

    def shared_step(
        self,
        batch: dict[str, dict],
        stage: Literal["train", "val", "test"],
    ) -> dict[str, torch.Tensor]:
        """Shared step for training/validation."""
        match self.trainer.ssl_phase:
            case "pretrain":
                return self.pretrain_step(batch, stage)
            case "probe" | "finetune":
                return self.probe_or_finetune_step(batch, stage)
            case _:
                msg = (
                    f"Invalid ssl phase {self.trainer.ssl_phase}."
                    " Expected 'pretrain' or 'probe' or 'finetune'"
                )
                raise ValueError(msg)

    def training_step(
        self,
        batch: dict[str, dict],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="train")

    def validation_step(
        self,
        batch: dict[str, dict],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="val")

    def test_step(
        self,
        batch: dict[str, dict],
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        return self.shared_step(batch, stage="test")

    def on_train_epoch_start(self) -> None:
        self.trainer.train_dataloader.set_epoch(self.trainer.current_epoch)

    def on_train_epoch_end(self) -> None:
        if self.ema_model is not None:
            self.update_ema()

    def update_ema(self) -> None:
        """Update EMA model."""
        momentum = 1 - 1 / (self.trainer.max_epochs * 0.2)
        for param, param_ema in zip(
            self.model.parameters(),
            self.ema_model.parameters(),
        ):
            param_ema.data.mul_(momentum).add_(
                (1.0 - momentum) * param.detach().to(param_ema.device)
            )
