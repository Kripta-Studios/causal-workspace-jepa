"""PointMass2D deterministic action-conditioned dynamics."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.base import SyntheticDataset


def step_pointmass(
    state: np.ndarray,
    action: np.ndarray,
    *,
    dt: float = 0.1,
    mass: float = 1.0,
    drag: float = 0.05,
    force_scale: float = 1.0,
) -> np.ndarray:
    next_state = state.astype(np.float64, copy=True)
    accel = np.clip(action, -1.0, 1.0) * force_scale / mass
    next_state[2:4] = (1.0 - drag * dt) * next_state[2:4] + accel * dt
    next_state[0:2] = next_state[0:2] + next_state[2:4] * dt
    return next_state.astype(np.float32)


def generate_pointmass2d(
    *,
    trajectories: int,
    steps: int,
    seed: int,
    mass: float = 1.0,
    drag: float = 0.05,
    force_scale: float = 1.0,
    observation_noise: float = 0.0,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    states = np.zeros((trajectories, steps, 4), dtype=np.float32)
    actions = np.zeros((trajectories, steps - 1, 2), dtype=np.float32)
    for trajectory in range(trajectories):
        state = np.array(
            [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-0.2, 0.2),
            ],
            dtype=np.float32,
        )
        states[trajectory, 0] = state
        for step in range(steps - 1):
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            actions[trajectory, step] = action
            state = step_pointmass(
                state,
                action,
                mass=mass,
                drag=drag,
                force_scale=force_scale,
            )
            states[trajectory, step + 1] = state
    observations = states.copy()
    if observation_noise:
        observations += rng.normal(0.0, observation_noise, size=observations.shape).astype(np.float32)
    return SyntheticDataset(
        name="pointmass2d",
        observations=observations.astype(np.float32),
        actions=actions,
        states=states,
        metadata={
            "env": "PointMass2D",
            "seed": seed,
            "mass": mass,
            "drag": drag,
            "force_scale": force_scale,
            "observation_noise": observation_noise,
            "state_names": ["x", "y", "vx", "vy"],
            "action_names": ["ax", "ay"],
        },
    )
