"""Closed-loop planning utilities."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.pointmass import step_pointmass


def pointmass_rollout_state(initial_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state = initial_state.astype(np.float32)
    for action in actions:
        state = step_pointmass(state, action)
    return state


def pointmass_position_mse(state: np.ndarray, goal: np.ndarray) -> float:
    return float(np.mean((state[:2] - np.asarray(goal, dtype=np.float32)) ** 2))


def pointmass_rollout_cost(initial_state: np.ndarray, actions: np.ndarray, goal: np.ndarray) -> float:
    state = pointmass_rollout_state(initial_state, actions)
    return float(np.sum((state[:2] - goal) ** 2))
