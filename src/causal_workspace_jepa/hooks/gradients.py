"""Finite-difference utilities for CPU-safe Jacobian checks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def finite_difference_jacobian(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    epsilon: float = 1e-4,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y0 = np.asarray(fn(x), dtype=np.float64).reshape(-1)
    jacobian = np.zeros((y0.size, x.size), dtype=np.float64)
    flat_x = x.reshape(-1)
    for index in range(flat_x.size):
        plus = flat_x.copy()
        minus = flat_x.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        y_plus = np.asarray(fn(plus.reshape(x.shape)), dtype=np.float64).reshape(-1)
        y_minus = np.asarray(fn(minus.reshape(x.shape)), dtype=np.float64).reshape(-1)
        jacobian[:, index] = (y_plus - y_minus) / (2.0 * epsilon)
    return jacobian.reshape((*y0.shape, *x.shape))
