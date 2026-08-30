"""Deterministic event-subset ranking from canonical event identity."""

from __future__ import annotations

import hashlib

import numpy as np


SUBSET_RANKING = "splitmix64_event_id_v1"
_MASK64 = (1 << 64) - 1


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Return stable 64-bit ranks for EVENT_ID values and one subset seed."""
    value = np.asarray(values, dtype=np.uint64) ^ np.uint64(int(seed) & _MASK64)
    with np.errstate(over="ignore"):
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        value = (value ^ (value >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return value ^ (value >> np.uint64(31))


def deterministic_subset_positions(
    event_ids: np.ndarray,
    count: int | None,
    seed: int,
) -> np.ndarray:
    """Select a sorted prefix of one deterministic EVENT_ID ranking."""
    event_ids = np.asarray(event_ids)
    if event_ids.ndim != 1:
        raise ValueError(f"EVENT_ID must be one-dimensional; shape={event_ids.shape}")
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("EVENT_ID values must be unique within each canonical split")
    if count is None or int(count) >= len(event_ids):
        return np.arange(len(event_ids), dtype=np.int64)
    if int(count) <= 0:
        raise ValueError("Requested event subset size must be positive")

    positions = np.arange(len(event_ids), dtype=np.int64)
    ranks = splitmix64(event_ids, seed)
    order = np.lexsort((positions, event_ids.astype(np.uint64), ranks))
    return np.sort(order[: int(count)].astype(np.int64, copy=False))


def event_id_digest(event_ids: np.ndarray) -> str:
    """Digest one ordered EVENT_ID sequence for artifact provenance."""
    values = np.asarray(event_ids, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()
