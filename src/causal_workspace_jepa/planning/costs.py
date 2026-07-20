"""Planner cost functions."""

from __future__ import annotations

import numpy as np


def squared_goal_cost(states: np.ndarray, goal: np.ndarray) -> np.ndarray:
    positions = states[..., :2]
    return np.sum((positions - goal) ** 2, axis=-1)
