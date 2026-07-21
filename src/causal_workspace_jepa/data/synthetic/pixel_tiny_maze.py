"""Deterministic pixel TinyMaze trajectories for small end-to-end JEPAs."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.base import SyntheticDataset
from causal_workspace_jepa.data.synthetic.tiny_maze import ACTIONS, GRID_SIZE, WALLS, step_tiny_maze


def valid_positions() -> tuple[tuple[int, int], ...]:
    """Return the fixed non-wall grid positions in lexical order."""

    return tuple(
        (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
        if (x, y) not in WALLS
    )


def render_pixel_tiny_maze(
    position: np.ndarray | tuple[int, int],
    goal: np.ndarray | tuple[int, int],
    *,
    cell_size: int = 4,
) -> np.ndarray:
    """Render walls, goal, and agent into separate float32 channels."""

    if cell_size < 1:
        raise ValueError("cell_size must be positive")
    position_tuple = tuple(int(value) for value in position)
    goal_tuple = tuple(int(value) for value in goal)
    if position_tuple not in valid_positions() or goal_tuple not in valid_positions():
        raise ValueError("position and goal must be non-wall cells")
    image = np.zeros((3, GRID_SIZE * cell_size, GRID_SIZE * cell_size), dtype=np.float32)
    for wall_x, wall_y in WALLS:
        image[
            0,
            wall_y * cell_size : (wall_y + 1) * cell_size,
            wall_x * cell_size : (wall_x + 1) * cell_size,
        ] = 1.0
    goal_x, goal_y = goal_tuple
    image[
        1,
        goal_y * cell_size : (goal_y + 1) * cell_size,
        goal_x * cell_size : (goal_x + 1) * cell_size,
    ] = 1.0
    x, y = position_tuple
    image[
        2,
        y * cell_size : (y + 1) * cell_size,
        x * cell_size : (x + 1) * cell_size,
    ] = 1.0
    return image


def generate_pixel_tiny_maze(
    *,
    trajectories: int,
    steps: int,
    seed: int,
    cell_size: int = 4,
) -> SyntheticDataset:
    """Generate image/action sequences with explicit state and split-safe seeds."""

    if trajectories < 1 or steps < 2:
        raise ValueError("trajectories must be positive and steps must be at least two")
    rng = np.random.default_rng(seed)
    positions = valid_positions()
    resolution = GRID_SIZE * cell_size
    observations = np.zeros((trajectories, steps, 3, resolution, resolution), dtype=np.float32)
    actions = np.zeros((trajectories, steps - 1, 4), dtype=np.float32)
    states = np.zeros((trajectories, steps, 5), dtype=np.float32)
    for trajectory in range(trajectories):
        goal = np.asarray(positions[int(rng.integers(0, len(positions)))], dtype=np.int64)
        position = np.asarray(positions[int(rng.integers(0, len(positions)))], dtype=np.int64)
        if np.array_equal(position, goal):
            position = np.asarray(positions[(positions.index(tuple(goal)) + 1) % len(positions)])
        for step in range(steps):
            observations[trajectory, step] = render_pixel_tiny_maze(
                position, goal, cell_size=cell_size
            )
            states[trajectory, step] = np.asarray(
                [position[0], position[1], goal[0], goal[1], np.abs(position - goal).sum()],
                dtype=np.float32,
            )
            if step < steps - 1:
                action_id = int(rng.integers(0, len(ACTIONS)))
                actions[trajectory, step, action_id] = 1.0
                position = step_tiny_maze(position, action_id)
    return SyntheticDataset(
        name="pixel_tiny_maze",
        observations=observations,
        actions=actions,
        states=states,
        metadata={
            "env": "PixelTinyMaze",
            "seed": seed,
            "grid_size": GRID_SIZE,
            "cell_size": cell_size,
            "resolution": resolution,
            "walls": sorted([list(wall) for wall in WALLS]),
            "state_names": ["x", "y", "goal_x", "goal_y", "manhattan_distance"],
            "action_names": ["up", "down", "left", "right"],
            "renderer_channels": ["walls", "goal", "agent"],
        },
    )
