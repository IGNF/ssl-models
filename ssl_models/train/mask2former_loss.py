"""Mask2Former-style Hungarian loss utilities for semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover - runtime dependency check
    linear_sum_assignment = None
    SCIPY_IMPORT_ERROR = exc
else:  # pragma: no cover - optional import branch
    SCIPY_IMPORT_ERROR = None


def _pairwise_dice_cost(mask_logits: Tensor, tgt_masks: Tensor, eps: float = 1.0) -> Tensor:
    """Compute pairwise Dice cost between query masks and target masks.

    Args:
        mask_logits: Query mask logits with shape (Q, P).
        tgt_masks: Target binary masks with shape (T, P).
    """
    mask_prob = mask_logits.sigmoid()
    intersection = 2 * torch.einsum("qp,tp->qt", mask_prob, tgt_masks)
    denominator = (
        mask_prob.sum(dim=1, keepdim=True)
        + tgt_masks.sum(dim=1).unsqueeze(0)
    )
    return 1 - (intersection + eps) / (denominator + eps)


def _pairwise_bce_cost(mask_logits: Tensor, tgt_masks: Tensor) -> Tensor:
    """Compute pairwise BCE cost between query masks and target masks."""
    q, _ = mask_logits.shape
    t, _ = tgt_masks.shape
    logits = mask_logits[:, None, :].expand(q, t, -1)
    targets = tgt_masks[None, :, :].expand(q, t, -1)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)


@dataclass
class HungarianLossConfig:
    """Config values for Hungarian matching loss."""

    cost_class: float = 2.0
    cost_mask: float = 5.0
    cost_dice: float = 5.0
    loss_class: float = 2.0
    loss_mask: float = 5.0
    loss_dice: float = 5.0
    eos_coef: float = 0.1
    num_points: int = 1024
    aux_weight: float = 1.0


class HungarianMatcher(nn.Module):
    """Hungarian matcher for mask classification outputs."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 1024,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        if linear_sum_assignment is None:
            msg = (
                "scipy is required for Hungarian matching. "
                f"Import error: {SCIPY_IMPORT_ERROR}"
            )
            raise ImportError(msg)

        pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
        pred_masks = outputs["pred_masks"]  # (B, Q, H, W)
        bs, num_queries, _ = pred_logits.shape

        out_prob = pred_logits.softmax(-1)
        indices: list[tuple[Tensor, Tensor]] = []

        for batch_idx in range(bs):
            tgt_ids = targets[batch_idx]["labels"]
            tgt_masks = targets[batch_idx]["masks"]

            if tgt_ids.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=pred_logits.device)
                indices.append((empty, empty))
                continue

            cost_class = -out_prob[batch_idx][:, tgt_ids]

            out_mask = pred_masks[batch_idx]
            if tgt_masks.shape[-2:] != out_mask.shape[-2:]:
                tgt_masks = F.interpolate(
                    tgt_masks.unsqueeze(1),
                    size=out_mask.shape[-2:],
                    mode="nearest",
                ).squeeze(1)

            out_mask = out_mask.flatten(1)
            tgt_masks = tgt_masks.flatten(1)

            if out_mask.shape[1] > self.num_points:
                point_inds = torch.randperm(out_mask.shape[1], device=out_mask.device)[: self.num_points]
                out_mask = out_mask[:, point_inds]
                tgt_masks = tgt_masks[:, point_inds]

            cost_mask = _pairwise_bce_cost(out_mask, tgt_masks)
            cost_dice = _pairwise_dice_cost(out_mask, tgt_masks)

            total_cost = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )
            row_ind, col_ind = linear_sum_assignment(total_cost.cpu().numpy())

            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64, device=pred_logits.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=pred_logits.device),
                )
            )

        assert len(indices) == bs
        assert all(src_idx.numel() <= num_queries for src_idx, _ in indices)
        return indices


class Mask2FormerHungarianCriterion(nn.Module):
    """Set-based Hungarian criterion for Mask2Former outputs."""

    def __init__(self, num_classes: int, cfg: HungarianLossConfig | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cfg = cfg or HungarianLossConfig()
        self.matcher = HungarianMatcher(
            cost_class=self.cfg.cost_class,
            cost_mask=self.cfg.cost_mask,
            cost_dice=self.cfg.cost_dice,
            num_points=self.cfg.num_points,
        )

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.cfg.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _loss_labels(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> Tensor:
        src_logits = outputs["pred_logits"]
        bs, num_queries, _ = src_logits.shape
        target_classes = torch.full(
            (bs, num_queries),
            fill_value=self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[batch_idx, src_idx] = targets[batch_idx]["labels"][tgt_idx]

        return F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight.to(src_logits.device),
        )

    def _loss_masks(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        pred_masks = outputs["pred_masks"]
        src_masks_list = []
        tgt_masks_list = []

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_masks = pred_masks[batch_idx, src_idx]
            tgt_masks = targets[batch_idx]["masks"][tgt_idx]
            if tgt_masks.shape[-2:] != src_masks.shape[-2:]:
                tgt_masks = F.interpolate(
                    tgt_masks.unsqueeze(1),
                    size=src_masks.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
            src_masks_list.append(src_masks)
            tgt_masks_list.append(tgt_masks)

        if not src_masks_list:
            zero = pred_masks.sum() * 0.0
            return zero, zero

        src_masks = torch.cat(src_masks_list, dim=0)
        tgt_masks = torch.cat(tgt_masks_list, dim=0)
        num_masks = max(tgt_masks.shape[0], 1)

        loss_mask = F.binary_cross_entropy_with_logits(src_masks, tgt_masks, reduction="none")
        loss_mask = loss_mask.mean(dim=(-2, -1)).sum() / num_masks

        src_prob = src_masks.sigmoid().flatten(1)
        tgt_flat = tgt_masks.flatten(1)
        numerator = 2 * (src_prob * tgt_flat).sum(dim=1)
        denominator = src_prob.sum(dim=1) + tgt_flat.sum(dim=1)
        loss_dice = (1 - (numerator + 1.0) / (denominator + 1.0)).sum() / num_masks

        return loss_mask, loss_dice

    def _compute_single(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> Tensor:
        indices = self.matcher(outputs, targets)
        loss_cls = self._loss_labels(outputs, targets, indices)
        loss_mask, loss_dice = self._loss_masks(outputs, targets, indices)
        return (
            self.cfg.loss_class * loss_cls
            + self.cfg.loss_mask * loss_mask
            + self.cfg.loss_dice * loss_dice
        )

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> Tensor:
        total_loss = self._compute_single(outputs, targets)
        for aux_out in outputs.get("aux_outputs", []):
            total_loss = total_loss + self.cfg.aux_weight * self._compute_single(aux_out, targets)
        return total_loss
