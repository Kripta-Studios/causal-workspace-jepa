"""World-model finite-difference lens helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from causal_workspace_jepa.hooks.gradients import finite_difference_jacobian


def local_lens(
    fn: Callable[[np.ndarray], np.ndarray],
    activation: np.ndarray,
    *,
    epsilon: float = 1e-4,
) -> np.ndarray:
    return finite_difference_jacobian(fn, activation, epsilon=epsilon)


def finite_difference_effect(
    fn: Callable[[np.ndarray], np.ndarray],
    activation: np.ndarray,
    direction: np.ndarray,
    *,
    magnitude: float,
) -> np.ndarray:
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return fn(activation + magnitude * direction) - fn(activation)
