"""Activation patching metrics."""

from __future__ import annotations

import numpy as np


def normalized_recovery(clean: np.ndarray, corrupted: np.ndarray, patched: np.ndarray) -> float:
    denominator = float(np.linalg.norm(clean - corrupted))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.linalg.norm(clean - patched) / denominator)
