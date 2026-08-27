"""Planner cost functions."""

from __future__ import annotations

import numpy as np


def squared_goal_cost(states: np.ndarray, goal: np.ndarray) -> np.ndarray:
    positions = states[..., :2]
    return np.sum((positions - goal) ** 2, axis=-1)


def squared_latent_goal_cost(predicted_latents: np.ndarray, z_goal: np.ndarray) -> np.ndarray:
    """Identical latent-goal cost for every MiniPush planner."""

    return np.sum((predicted_latents - z_goal) ** 2, axis=-1)
