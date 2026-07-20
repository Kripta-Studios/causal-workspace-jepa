"""BouncingBall2D deterministic dynamics."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.base import SyntheticDataset


def generate_bouncing_ball2d(
    *,
    trajectories: int,
    steps: int,
    seed: int,
    gravity: float = -9.8,
    restitution: float = 0.85,
    dt: float = 0.05,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    states = np.zeros((trajectories, steps, 4), dtype=np.float32)
    actions = np.zeros((trajectories, steps - 1, 1), dtype=np.float32)
    for trajectory in range(trajectories):
        state = np.array(
            [rng.uniform(0.2, 0.8), rng.uniform(0.4, 0.9), rng.uniform(-0.5, 0.5), 0.0],
            dtype=np.float32,
        )
        states[trajectory, 0] = state
        for step in range(steps - 1):
            horizontal = float(rng.uniform(-1.0, 1.0))
            actions[trajectory, step, 0] = horizontal
            state = state.copy()
            state[2] += 0.2 * horizontal * dt
            state[3] += gravity * dt
            state[0] += state[2] * dt
            state[1] += state[3] * dt
            if state[0] < 0.0 or state[0] > 1.0:
                state[0] = np.clip(state[0], 0.0, 1.0)
                state[2] = -restitution * state[2]
            if state[1] < 0.0:
                state[1] = 0.0
                state[3] = -restitution * state[3]
            if state[1] > 1.0:
                state[1] = 1.0
                state[3] = -restitution * state[3]
            states[trajectory, step + 1] = state
    return SyntheticDataset(
        name="bouncing_ball2d",
        observations=states.copy(),
        actions=actions,
        states=states,
        metadata={
            "env": "BouncingBall2D",
            "seed": seed,
            "gravity": gravity,
            "restitution": restitution,
            "state_names": ["x", "y", "vx", "vy"],
            "action_names": ["horizontal_accel"],
        },
    )
