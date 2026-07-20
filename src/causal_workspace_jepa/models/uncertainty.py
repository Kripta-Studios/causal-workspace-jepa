"""CPU-safe uncertainty metrics."""

from __future__ import annotations

import numpy as np


def ensemble_disagreement(predictions: np.ndarray) -> np.ndarray:
    return np.var(predictions, axis=0)
