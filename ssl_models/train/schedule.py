"""Step-based schedules (lr, weight decay, DINO teacher temperature and momentum)."""

import math
from typing import Literal


def cosine(start: float, end: float, step: int, total: int) -> float:
    """Cosine interpolation from start (step 0) to end (step total)."""
    if total <= 0:
        return end
    t = min(max(step / total, 0.0), 1.0)
    return end + (start - end) * (1 + math.cos(math.pi * t)) / 2


def linear_warmup(start: float, end: float, step: int, warmup: int) -> float:
    """Linear interpolation from start to end over warmup steps, then end."""
    if warmup <= 0 or step >= warmup:
        return end
    return start + (end - start) * step / warmup


def lr_factor(
    step: int,
    total: int,
    warmup_pct: float,
    schedule: Literal["cosine", "constant"],
    final_ratio: float,
    freeze_pct: float = 0.0,
) -> float:
    """Learning rate multiplier: linear warmup, then cosine decay or constant.

    The multiplier is 0 over the first `freeze_pct` steps (frozen params).
    """
    warmup = round(warmup_pct * total)

    if step < round(freeze_pct * total):
        return 0.0
    if step < warmup:
        return (step + 1) / warmup
    match schedule:
        case "cosine":
            return cosine(1.0, final_ratio, step - warmup, total - warmup)
        case "constant":
            return 1.0
        case _:
            msg = f"Invalid lr schedule {schedule}. Expected 'cosine' or 'constant'."
            raise ValueError(msg)
