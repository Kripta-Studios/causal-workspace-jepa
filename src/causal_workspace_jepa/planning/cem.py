"""Tiny random-shooting/CEM-style planner for smoke tests."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.common.types import LatentState
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA
from causal_workspace_jepa.planning.costs import squared_goal_cost


def random_shooting_plan(
    model: TinyActionConditionedJEPA,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    candidates: int,
    seed: int,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    action_dim = model.config.action_dim
    action_sequences = rng.uniform(
        action_low,
        action_high,
        size=(candidates, horizon, action_dim),
    ).astype(np.float32)
    repeated_latent = np.repeat(model.encode(observation[None, :]).tensor, candidates, axis=0)
    output = model.predict(LatentState(repeated_latent), action_sequences, return_intermediates=False)
    assert output.decoded_state is not None
    decoded = output.decoded_state["state"]
    costs = squared_goal_cost(decoded[:, -1, :], goal)
    best = int(np.argmin(costs))
    return {
        "actions": action_sequences[best],
        "first_action": action_sequences[best, 0],
        "predicted_cost": float(costs[best]),
        "random_mean_cost": float(np.mean(costs)),
    }
