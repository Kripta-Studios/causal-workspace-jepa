"""TwoBodyCollision deterministic toy interactions."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.base import SyntheticDataset


def _resolve_collision(state: np.ndarray, restitution: float, radius: float) -> tuple[np.ndarray, float]:
    pos1 = state[0:2]
    vel1 = state[2:4]
    mass1 = float(state[4])
    pos2 = state[5:7]
    vel2 = state[7:9]
    mass2 = float(state[9])
    delta = pos2 - pos1
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6 or distance > 2.0 * radius:
        return state, 0.0
    normal = delta / distance
    rel_vel = vel2 - vel1
    vel_along_normal = float(np.dot(rel_vel, normal))
    if vel_along_normal > 0:
        return state, 1.0
    impulse = -(1.0 + restitution) * vel_along_normal / (1.0 / mass1 + 1.0 / mass2)
    vel1 = vel1 - (impulse / mass1) * normal
    vel2 = vel2 + (impulse / mass2) * normal
    corrected = state.copy()
    corrected[2:4] = vel1
    corrected[7:9] = vel2
    return corrected, 1.0


def generate_two_body_collision(
    *,
    trajectories: int,
    steps: int,
    seed: int,
    restitution: float = 0.9,
    dt: float = 0.05,
    radius: float = 0.08,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    states = np.zeros((trajectories, steps, 11), dtype=np.float32)
    actions = np.zeros((trajectories, steps - 1, 2), dtype=np.float32)
    for trajectory in range(trajectories):
        state = np.array(
            [
                rng.uniform(0.15, 0.35),
                rng.uniform(0.3, 0.7),
                rng.uniform(0.2, 0.7),
                rng.uniform(-0.15, 0.15),
                rng.uniform(0.7, 1.5),
                rng.uniform(0.65, 0.85),
                rng.uniform(0.3, 0.7),
                rng.uniform(-0.6, -0.2),
                rng.uniform(-0.15, 0.15),
                rng.uniform(0.7, 1.5),
                0.0,
            ],
            dtype=np.float32,
        )
        states[trajectory, 0] = state
        for step in range(steps - 1):
            action = rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
            actions[trajectory, step] = action
            state = state.copy()
            state[2:4] += action * dt
            state[0:2] += state[2:4] * dt
            state[5:7] += state[7:9] * dt
            for offset in (0, 5):
                for dim in (0, 1):
                    index = offset + dim
                    velocity = offset + 2 + dim
                    if state[index] < radius or state[index] > 1.0 - radius:
                        state[index] = np.clip(state[index], radius, 1.0 - radius)
                        state[velocity] = -restitution * state[velocity]
            state, contact = _resolve_collision(state, restitution, radius)
            state[10] = contact
            states[trajectory, step + 1] = state
    return SyntheticDataset(
        name="two_body_collision",
        observations=states.copy(),
        actions=actions,
        states=states,
        metadata={
            "env": "TwoBodyCollision",
            "seed": seed,
            "restitution": restitution,
            "radius": radius,
            "state_names": [
                "x1",
                "y1",
                "vx1",
                "vy1",
                "mass1",
                "x2",
                "y2",
                "vx2",
                "vy2",
                "mass2",
                "contact",
            ],
            "action_names": ["force1_x", "force1_y"],
        },
    )
