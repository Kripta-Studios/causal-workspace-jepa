"""Deterministic split helpers."""

from __future__ import annotations

import numpy as np


def deterministic_split_ids(count: int, seed: int, train_fraction: float = 0.7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.arange(count)
    rng.shuffle(ids)
    train_end = int(count * train_fraction)
    val_end = train_end + int((count - train_end) / 2)
    return {
        "train": np.sort(ids[:train_end]),
        "validation": np.sort(ids[train_end:val_end]),
        "test": np.sort(ids[val_end:]),
    }
