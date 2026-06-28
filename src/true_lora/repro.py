from __future__ import annotations

import random

import torch


def set_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
