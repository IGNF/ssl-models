"""DINO, iBOT and KoLeo losses of DINOv3-style pretraining.

The teacher targets are centered by Sinkhorn-Knopp, distributed across processes.
Losses and targets are computed in float32, outside of autocast.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def _all_reduce(x: Tensor) -> Tensor:
    """Sum over processes, in place, if distributed."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(x)
    return x


@torch.no_grad()
def sinkhorn_knopp(logits: Tensor, temp: float, num_iter: int = 3) -> Tensor:
    """Teacher targets with equipartition over prototypes, across processes.

    Args:
        logits: [N, K] teacher cosine logits in [-1, 1]. N may differ between
            processes (e.g. masked tokens), and may be 0. Every process must call
            this function the same number of times.
        temp: teacher temperature.
        num_iter: number of Sinkhorn-Knopp iterations.

    Returns:
        [N, K] target probabilities (rows sum to 1).

    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        K = logits.shape[1]  # noqa: N806
        num_total = _all_reduce(
            torch.tensor(float(logits.shape[0]), device=logits.device),
        )
        # logits <= 1, shift for numerical stability (cancels out)
        Q = torch.exp((logits - 1) / temp).t()  # noqa: N806 [K, N]
        Q /= _all_reduce(Q.sum()).clamp(min=1e-30)  # noqa: N806
        for _ in range(num_iter):
            Q /= _all_reduce(Q.sum(dim=1, keepdim=True)).clamp(min=1e-30)  # noqa: N806
            Q /= K  # noqa: N806
            Q /= Q.sum(dim=0, keepdim=True).clamp(min=1e-30)  # noqa: N806
            Q /= num_total.clamp(min=1)  # noqa: N806
        Q *= num_total  # noqa: N806
        return Q.t()


def dino_loss(
    student_global: Tensor,
    student_local: Tensor | None,
    teacher_probs: Tensor,
    student_temp: float,
    num_global: int,
    num_local: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """DINO cross-entropy between student crops and teacher global crops.

    The crops of the B samples are stacked sample-major ([B * n, ...]).

    Args:
        student_global: [B * num_global, K] student logits of global crops.
        student_local: [B * num_local, K] student logits of local crops, or None.
        teacher_probs: [B * num_global, K] teacher targets of global crops.
        student_temp: student temperature.
        num_global: number of global crops per sample.
        num_local: number of local crops per sample.

    Returns:
        The mean over all pairs, over pairs of distinct global crops (0 if
        num_global=1), and over (local, global) pairs (0 if no local crop).

    """
    with torch.autocast(device_type=teacher_probs.device.type, enabled=False):
        teacher_probs = teacher_probs.float().unflatten(0, (-1, num_global))

        def cross_entropy(student: Tensor) -> Tensor:
            log_probs = F.log_softmax(student.float() / student_temp, dim=-1)
            # [num_student, num_teacher] mean cross-entropy over the batch
            return (
                -torch.einsum("bsk,btk->st", log_probs, teacher_probs)
                / student.shape[0]
            )

        ce_global = cross_entropy(student_global.unflatten(0, (-1, num_global)))
        off_diag = ~torch.eye(num_global, dtype=torch.bool, device=ce_global.device)
        num_global_pairs = int(off_diag.sum())
        loss_global = (
            ce_global[off_diag].mean() if num_global_pairs else ce_global.sum() * 0
        )

        if student_local is not None:
            ce_local = cross_entropy(student_local.unflatten(0, (-1, num_local)))
            num_local_pairs = ce_local.numel()
            loss_local = ce_local.mean()
        else:
            num_local_pairs = 0
            loss_local = ce_global.sum() * 0

        num_pairs = max(num_global_pairs + num_local_pairs, 1)
        loss = (
            num_global_pairs * loss_global + num_local_pairs * loss_local
        ) / num_pairs
    return loss, loss_global, loss_local


def ibot_loss(
    student_logits: Tensor,
    weights: Tensor,
    teacher_probs: Tensor,
    student_temp: float,
) -> Tensor:
    """iBOT cross-entropy on masked tokens, weighted mean over the tokens.

    With weights 1 / number of masked tokens of their image, each image has the
    same total weight, and the loss is the mean over the images that have at
    least one masked token.

    Args:
        student_logits: [M, K] student logits of masked tokens.
        weights: [M] weights of the tokens.
        teacher_probs: [M, K] teacher targets of the same tokens.
        student_temp: student temperature.

    Returns:
        The weighted mean cross-entropy (0 if there is no masked token).

    """
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        log_probs = F.log_softmax(student_logits.float() / student_temp, dim=-1)
        ce = -(teacher_probs.float() * log_probs).sum(dim=-1)  # [M]
        return (ce * weights).sum() / weights.sum().clamp(min=1e-12)


def koleo_loss(x: Tensor, num_global: int, eps: float = 1e-8) -> Tensor:
    """KoLeo regularizer: -mean log distance to the nearest neighbor in the batch,
    within each global crop.

    Args:
        x: [B * num_global, C] student embeddings of global crops, stacked
            sample-major.
        num_global: number of global crops per sample.
        eps: for the normalization and the log.

    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = F.normalize(x.float(), dim=-1, eps=eps)
        x = x.unflatten(0, (-1, num_global)).transpose(0, 1)  # [num_global, B, C]
        if x.shape[1] < 2:  # noqa: PLR2004
            return x.sum() * 0
        with torch.no_grad():
            dots = x @ x.transpose(1, 2)
            dots.diagonal(dim1=1, dim2=2).fill_(-2.0)
            idx = dots.argmax(dim=2, keepdim=True)
        dist_nn = (x - x.gather(1, idx.expand_as(x))).norm(dim=-1)
        return -torch.log(dist_nn + eps).mean()
