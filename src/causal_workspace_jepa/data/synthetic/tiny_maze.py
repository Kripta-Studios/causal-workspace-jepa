"""TinyMaze deterministic navigation data."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.base import SyntheticDataset

GRID_SIZE = 5
WALLS = {(2, 1), (2, 2), (2, 3)}
ACTIONS = {
    0: np.array([0, -1], dtype=np.int64),
    1: np.array([0, 1], dtype=np.int64),
    2: np.array([-1, 0], dtype=np.int64),
    3: np.array([1, 0], dtype=np.int64),
}


def step_tiny_maze(position: np.ndarray, action_id: int) -> np.ndarray:
    proposed = position + ACTIONS[int(action_id)]
    proposed = np.clip(proposed, 0, GRID_SIZE - 1)
    if tuple(int(v) for v in proposed) in WALLS:
        return position.copy()
    return proposed.astype(np.int64)


def generate_tiny_maze(
    *,
    trajectories: int,
    steps: int,
    seed: int,
    goal: tuple[int, int] = (4, 4),
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    states = np.zeros((trajectories, steps, 5), dtype=np.float32)
    actions = np.zeros((trajectories, steps - 1, 1), dtype=np.int64)
    goal_array = np.array(goal, dtype=np.int64)
    valid_positions = [
        np.array([x, y], dtype=np.int64)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
        if (x, y) not in WALLS and (x, y) != goal
    ]
    for trajectory in range(trajectories):
        position = valid_positions[int(rng.integers(0, len(valid_positions)))].copy()
        for step in range(steps):
            distance = np.abs(position - goal_array).sum()
            states[trajectory, step] = np.array(
                [position[0], position[1], goal_array[0], goal_array[1], distance],
                dtype=np.float32,
            )
            if step < steps - 1:
                action_id = int(rng.integers(0, 4))
                actions[trajectory, step, 0] = action_id
                position = step_tiny_maze(position, action_id)
    observations = states / np.array([4.0, 4.0, 4.0, 4.0, 8.0], dtype=np.float32)
    return SyntheticDataset(
        name="tiny_maze",
        observations=observations.astype(np.float32),
        actions=actions,
        states=states,
        metadata={
            "env": "TinyMaze",
            "seed": seed,
            "grid_size": GRID_SIZE,
            "walls": sorted([list(wall) for wall in WALLS]),
            "goal": list(goal),
            "state_names": ["x", "y", "goal_x", "goal_y", "manhattan_distance"],
            "action_names": ["up", "down", "left", "right"],
        },
    )
