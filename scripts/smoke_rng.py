"""Deterministic random-state setup for isolated smoke modes."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_smoke(seed: int) -> None:
    """Reset all smoke RNGs so each mode is independent of call order."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
