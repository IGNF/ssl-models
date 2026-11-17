"""DINO, iBOT and KoLeo losses, with distributed Sinkhorn-Knopp centering.

All losses are computed in float32, outside of autocast.
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
def sinkhorn_knopp(logits: Tensor, temp: float, n_iter: int = 3) -> Tensor:
    """Teacher targets with equipartition over prototypes, across processes.

    Args:
        logits: [N, K] teacher cosine logits in [-1, 1]. N may differ between
            processes (e.g. masked tokens), and may be 0. Every process must call
            this function the same number of times.
        temp: teacher temperature.
        n_iter: number of Sinkhorn-Knopp iterations.

    Returns:
        [N, K] target probabilities (rows sum to 1).

    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        K = logits.shape[1]  # noqa: N806
        n_total = _all_reduce(
            torch.tensor(float(logits.shape[0]), device=logits.device),
        )
        # logits <= 1, shift for numerical stability (cancels out)
        Q = torch.exp((logits - 1) / temp).t()  # noqa: N806 [K, N]
        Q /= _all_reduce(Q.sum()).clamp(min=1e-30)  # noqa: N806
        for _ in range(n_iter):
            Q /= _all_reduce(Q.sum(dim=1, keepdim=True)).clamp(min=1e-30)  # noqa: N806
            Q /= K  # noqa: N806
            Q /= Q.sum(dim=0, keepdim=True).clamp(min=1e-30)  # noqa: N806
            Q /= n_total.clamp(min=1)  # noqa: N806
        Q *= n_total  # noqa: N806
        return Q.t()


def dino_loss(
    student_global: Tensor,
    student_local: Tensor | None,
    teacher_probs: Tensor,
    student_temp: float,
) -> dict[str, Tensor]:
    """DINO cross-entropy between student crops and teacher global crops.

    Args:
        student_global: [n_global, B, K] student logits of global crops.
        student_local: [n_local, B, K] student logits of local crops, or None.
        teacher_probs: [n_global, B, K] teacher targets of global crops.
        student_temp: student temperature.

    Returns:
        "global": mean over pairs of distinct global crops (0 if n_global=1),
        "local": mean over (local, global) pairs (0 if no local crop),
        "total": mean over all pairs.

    """
    with torch.autocast(device_type=teacher_probs.device.type, enabled=False):
        teacher_probs = teacher_probs.float()
        n_global = student_global.shape[0]

        def cross_entropy(student: Tensor) -> Tensor:
            log_probs = F.log_softmax(student.float() / student_temp, dim=-1)
            # [n_student, n_teacher] mean cross-entropy over the batch
            return -torch.einsum("sbk,tbk->st", log_probs, teacher_probs) / (
                student.shape[1]
            )

        ce_global = cross_entropy(student_global)
        off_diag = ~torch.eye(n_global, dtype=torch.bool, device=ce_global.device)
        n_global_pairs = int(off_diag.sum())
        loss_global = (
            ce_global[off_diag].mean() if n_global_pairs else ce_global.sum() * 0
        )

        if student_local is not None and student_local.shape[0]:
            ce_local = cross_entropy(student_local)
            n_local_pairs = ce_local.numel()
            loss_local = ce_local.mean()
        else:
            n_local_pairs = 0
            loss_local = ce_global.sum() * 0

        n_pairs = max(n_global_pairs + n_local_pairs, 1)
        loss_total = (
            n_global_pairs * loss_global + n_local_pairs * loss_local
        ) / n_pairs
    return {"global": loss_global, "local": loss_local, "total": loss_total}


def ibot_loss(
    student_logits: Tensor,
    teacher_probs: Tensor,
    num_masked: Tensor,
    student_temp: float,
) -> Tensor:
    """iBOT cross-entropy on masked tokens.

    Each image has the same total weight, whatever its number of masked tokens;
    the loss is averaged over the images that have at least one masked token.

    Args:
        student_logits: [M, K] student logits of masked tokens, grouped by image.
        teacher_probs: [M, K] teacher targets of the same tokens.
        num_masked: [N] number of masked tokens per image (sums to M).
        student_temp: student temperature.

    """
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        log_probs = F.log_softmax(student_logits.float() / student_temp, dim=-1)
        ce = -(teacher_probs.float() * log_probs).sum(dim=-1)  # [M]
        weights = (1.0 / num_masked.clamp(min=1).float()).repeat_interleave(
            num_masked,
        )
        n_images = (num_masked > 0).sum().clamp(min=1)
        return (ce * weights).sum() / n_images


def koleo_loss(x: Tensor, eps: float = 1e-8) -> Tensor:
    """KoLeo regularizer: -mean log distance to nearest neighbor in the batch."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = F.normalize(x.float(), dim=-1, eps=eps)
        if x.shape[0] < 2:  # noqa: PLR2004
            return x.sum() * 0
        with torch.no_grad():
            dots = x @ x.t()
            dots.fill_diagonal_(-2.0)
            idx = dots.argmax(dim=1)
        dist_nn = (x - x[idx]).norm(dim=-1)
        return -torch.log(dist_nn + eps).mean()


@torch.no_grad()
def teacher_stats(probs: Tensor) -> dict[str, Tensor]:
    """Collapse diagnostics of teacher targets [N, K].

    "entropy": mean entropy of targets (0 = one-hot, log K = uniform),
    "kl_uniform": KL of the mean target to uniform (0 = all prototypes used).
    """
    probs = probs.float()
    K = probs.shape[1]  # noqa: N806
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1).mean()
    mean = probs.mean(dim=0)
    kl_uniform = (mean * (mean.clamp(min=1e-12) * K).log()).sum()
    return {"entropy": entropy, "kl_uniform": kl_uniform}
