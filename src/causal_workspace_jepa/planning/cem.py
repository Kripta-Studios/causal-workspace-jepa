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
        "candidates_evaluated": int(candidates),
        "iterations": 1,
    }


def iterative_cem_plan(
    model: TinyActionConditionedJEPA,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    candidates: int,
    iterations: int,
    elite_fraction: float,
    seed: int,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> dict[str, np.ndarray | float | int]:
    """Short-horizon CEM. This is iterative search inside one replan, not amortized planning."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    rng = np.random.default_rng(seed)
    action_dim = model.config.action_dim
    mean = np.zeros((horizon, action_dim), dtype=np.float32)
    std = np.full((horizon, action_dim), 0.5 * (action_high - action_low), dtype=np.float32)
    start_latent = np.repeat(model.encode(observation[None, :]).tensor, candidates, axis=0)
    best_actions = mean.copy()
    best_cost = float("inf")
    evaluated = 0
    for _ in range(int(iterations)):
        noise = rng.normal(size=(candidates, horizon, action_dim)).astype(np.float32)
        action_sequences = np.clip(mean + std * noise, action_low, action_high)
        output = model.predict(LatentState(start_latent), action_sequences, return_intermediates=False)
        assert output.decoded_state is not None
        costs = squared_goal_cost(output.decoded_state["state"][:, -1, :], goal)
        evaluated += int(candidates)
        order = np.argsort(costs)
        elite_count = max(1, int(np.ceil(elite_fraction * candidates)))
        elite = action_sequences[order[:elite_count]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 1e-3)
        if float(costs[order[0]]) < best_cost:
            best_cost = float(costs[order[0]])
            best_actions = action_sequences[order[0]]
    return {
        "actions": best_actions,
        "first_action": best_actions[0],
        "predicted_cost": best_cost,
        "candidates_evaluated": evaluated,
        "iterations": int(iterations),
    }
