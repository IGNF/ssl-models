"""Step-based schedules for DINO pretraining."""

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
    warmup: int,
    schedule: Literal["cosine", "constant"],
    final_ratio: float,
) -> float:
    """Learning rate multiplier: linear warmup, then cosine decay or constant."""
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
