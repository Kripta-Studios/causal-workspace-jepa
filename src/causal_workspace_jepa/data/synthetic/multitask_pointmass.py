"""Goal- and dynamics-conditioned deterministic PointMass2D tasks."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.base import SyntheticDataset
from causal_workspace_jepa.data.synthetic.pointmass import step_pointmass


DEFAULT_GOALS = np.array(
    [[-0.8, -0.8], [-0.8, 0.8], [0.8, -0.8], [0.8, 0.8]],
    dtype=np.float32,
)


def generate_multitask_pointmass2d(
    *,
    trajectories: int,
    steps: int,
    seed: int,
    masses: tuple[float, float] = (0.4, 1.6),
    goals: np.ndarray = DEFAULT_GOALS,
) -> SyntheticDataset:
    """Generate balanced goal/mass tasks with observations ``[state, goal, mode]``."""

    if trajectories < len(goals) * len(masses):
        raise ValueError("trajectories must cover every goal/mass combination")
    if steps < 2:
        raise ValueError("steps must be at least two")
    rng = np.random.default_rng(seed)
    combinations = [
        (goal_id, mode)
        for mode in range(len(masses))
        for goal_id in range(len(goals))
    ]
    assignments = [combinations[index % len(combinations)] for index in range(trajectories)]
    rng.shuffle(assignments)
    observations = np.zeros((trajectories, steps, 8), dtype=np.float32)
    actions = np.zeros((trajectories, steps - 1, 2), dtype=np.float32)
    for trajectory, (goal_id, mode) in enumerate(assignments):
        state = np.array(
            [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
            ],
            dtype=np.float32,
        )
        context = np.concatenate(
            [goals[goal_id], np.eye(len(masses), dtype=np.float32)[mode]],
        )
        observations[trajectory, 0] = np.concatenate([state, context])
        for step in range(steps - 1):
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            actions[trajectory, step] = action
            state = step_pointmass(
                state,
                action,
                dt=0.25,
                mass=masses[mode],
                drag=0.08,
            )
            observations[trajectory, step + 1] = np.concatenate([state, context])
    return SyntheticDataset(
        name="multitask_pointmass2d",
        observations=observations,
        actions=actions,
        states=observations.copy(),
        metadata={
            "env": "MultiTaskPointMass2D",
            "seed": seed,
            "masses": list(masses),
            "goals": goals.tolist(),
            "state_names": [
                "x",
                "y",
                "vx",
                "vy",
                "goal_x",
                "goal_y",
                "mode_0",
                "mode_1",
            ],
            "action_names": ["ax", "ay"],
        },
    )
