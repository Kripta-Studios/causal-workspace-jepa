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


def deterministic_named_split_ids(
    count: int,
    seed: int,
    *,
    train: int,
    development: int,
    confirmation: int,
) -> dict[str, np.ndarray]:
    """Train/development/confirmation split that never names a protected `test` split."""

    if train + development + confirmation != int(count):
        raise ValueError("split sizes must sum to trajectory count")
    if min(train, development, confirmation) < 1:
        raise ValueError("each named split must be non-empty")
    rng = np.random.default_rng(seed)
    ids = np.arange(count)
    rng.shuffle(ids)
    train_ids = np.sort(ids[:train])
    development_ids = np.sort(ids[train : train + development])
    confirmation_ids = np.sort(ids[train + development :])
    return {
        "train": train_ids,
        "development": development_ids,
        "confirmation": confirmation_ids,
    }
