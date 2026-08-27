"""CPU-scale amortized latent interpolator. Not a LeFlow paper reproduction."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.common.types import LatentState
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA
from causal_workspace_jepa.planning.costs import squared_goal_cost


def _with_bias(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array, np.ones((*array.shape[:-1], 1), dtype=array.dtype)], axis=-1)


def _ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    xtx = design.T @ design
    regularizer = ridge * np.eye(xtx.shape[0], dtype=np.float32)
    return np.linalg.solve(xtx + regularizer, design.T @ target).astype(np.float32)


def fit_inverse_dynamics(
    latents: np.ndarray,
    actions: np.ndarray,
    *,
    include_delta: bool,
    ridge: float,
) -> np.ndarray:
    """Capacity-matched ridge: the no-delta arm pads with zeros of size latent_dim."""

    z_t = latents[:, :-1, :].reshape(-1, latents.shape[-1])
    z_next = latents[:, 1:, :].reshape(-1, latents.shape[-1])
    delta = z_next - z_t
    if include_delta:
        features = np.concatenate([z_t, z_next, delta], axis=-1)
    else:
        features = np.concatenate([z_t, z_next, np.zeros_like(delta)], axis=-1)
    return _ridge(_with_bias(features), actions.reshape(-1, actions.shape[-1]), ridge)


def decode_actions(
    weights: np.ndarray,
    z_t: np.ndarray,
    z_next: np.ndarray,
    *,
    include_delta: bool,
) -> np.ndarray:
    delta = (z_next - z_t) if include_delta else np.zeros_like(z_t)
    features = _with_bias(np.concatenate([z_t, z_next, delta], axis=-1))
    return features @ weights


def interpolate_latents(start: np.ndarray, goal: np.ndarray, horizon: int) -> np.ndarray:
    weights = np.linspace(0.0, 1.0, horizon + 1, dtype=np.float32)[:, None]
    return ((1.0 - weights) * start[None, :] + weights * goal[None, :]).astype(np.float32)


def amortized_latent_plan(
    model: TinyActionConditionedJEPA,
    inverse_weights: np.ndarray,
    observation: np.ndarray,
    goal_observation: np.ndarray,
    *,
    horizon: int,
    candidates: int,
    seed: int,
    noise_std: float,
    include_delta: bool = True,
) -> dict[str, np.ndarray | float | int]:
    rng = np.random.default_rng(seed)
    z0 = model.encode(observation[None, :]).tensor[0]
    zg = model.encode(goal_observation[None, :]).tensor[0]
    base = interpolate_latents(z0, zg, horizon)
    paths = np.repeat(base[None, :, :], candidates, axis=0)
    if candidates > 1 and noise_std > 0:
        noise = rng.normal(scale=noise_std, size=paths.shape).astype(np.float32)
        noise[:, 0, :] = 0
        noise[:, -1, :] = 0
        paths = paths + noise
    action_dim = model.config.action_dim
    action_sequences = np.zeros((candidates, horizon, action_dim), dtype=np.float32)
    for index in range(horizon):
        action_sequences[:, index, :] = decode_actions(
            inverse_weights,
            paths[:, index, :],
            paths[:, index + 1, :],
            include_delta=include_delta,
        )
    action_sequences = np.clip(action_sequences, -1.0, 1.0)
    start = np.repeat(z0[None, :], candidates, axis=0)
    rollout = model.predict(LatentState(start), action_sequences, return_intermediates=False)
    assert rollout.decoded_state is not None
    costs = squared_goal_cost(rollout.decoded_state["state"][:, -1, :], goal_observation[:2])
    best = int(np.argmin(costs))
    return {
        "actions": action_sequences[best],
        "first_action": action_sequences[best, 0],
        "predicted_cost": float(costs[best]),
        "candidates_evaluated": int(candidates),
        "iterations": 1,
    }


def action_flow_plan(
    model: TinyActionConditionedJEPA,
    action_weights: np.ndarray,
    observation: np.ndarray,
    goal_observation: np.ndarray,
    *,
    horizon: int,
) -> dict[str, np.ndarray | float | int]:
    z0 = model.encode(observation[None, :]).tensor
    zg = model.encode(goal_observation[None, :]).tensor
    features = _with_bias(np.concatenate([z0, zg], axis=-1))
    flat = features @ action_weights
    actions = np.clip(flat.reshape(horizon, model.config.action_dim), -1.0, 1.0)
    return {
        "actions": actions.astype(np.float32),
        "first_action": actions[0],
        "predicted_cost": float("nan"),
        "candidates_evaluated": 1,
        "iterations": 1,
    }


def fit_action_flow(
    latents: np.ndarray,
    actions: np.ndarray,
    *,
    horizon: int,
    ridge: float,
) -> np.ndarray:
    starts = latents[:, 0, :]
    goals = latents[:, horizon, :]
    seq = actions[:, :horizon, :].reshape(actions.shape[0], -1)
    features = _with_bias(np.concatenate([starts, goals], axis=-1))
    return _ridge(features, seq, ridge)
