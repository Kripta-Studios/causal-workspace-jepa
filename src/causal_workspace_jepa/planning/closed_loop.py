"""Closed-loop planning utilities."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.pointmass import step_pointmass


def pointmass_rollout_cost(initial_state: np.ndarray, actions: np.ndarray, goal: np.ndarray) -> float:
    state = initial_state.astype(np.float32)
    for action in actions:
        state = step_pointmass(state, action)
    return float(np.sum((state[:2] - goal) ** 2))
