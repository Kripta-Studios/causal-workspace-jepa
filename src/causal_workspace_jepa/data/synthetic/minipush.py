"""MiniPush 32x32 pixel smoke generator with masks."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.base import SyntheticDataset


def _draw_square(image: np.ndarray, mask: np.ndarray, x: int, y: int, value: int) -> None:
    image[max(0, y - 1) : min(image.shape[0], y + 2), max(0, x - 1) : min(image.shape[1], x + 2)] = value
    mask[max(0, y - 1) : min(mask.shape[0], y + 2), max(0, x - 1) : min(mask.shape[1], x + 2)] = 1


def generate_minipush(
    *,
    trajectories: int,
    steps: int,
    seed: int,
    resolution: int = 32,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    states = np.zeros((trajectories, steps, 6), dtype=np.float32)
    actions = np.zeros((trajectories, steps - 1, 2), dtype=np.int64)
    observations = np.zeros((trajectories, steps, resolution, resolution), dtype=np.uint8)
    masks = np.zeros((trajectories, steps, 3, resolution, resolution), dtype=np.uint8)
    for trajectory in range(trajectories):
        agent = rng.integers(4, resolution - 4, size=2)
        obj = rng.integers(8, resolution - 8, size=2)
        goal = rng.integers(6, resolution - 6, size=2)
        for step in range(steps):
            states[trajectory, step] = np.array([*agent, *obj, *goal], dtype=np.float32)
            _draw_square(observations[trajectory, step], masks[trajectory, step, 0], goal[0], goal[1], 80)
            _draw_square(observations[trajectory, step], masks[trajectory, step, 1], obj[0], obj[1], 160)
            _draw_square(
                observations[trajectory, step],
                masks[trajectory, step, 2],
                agent[0],
                agent[1],
                255,
            )
            if step < steps - 1:
                delta = rng.choice(np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64))
                actions[trajectory, step] = delta
                proposed = np.clip(agent + delta, 1, resolution - 2)
                if np.max(np.abs(proposed - obj)) <= 1:
                    obj = np.clip(obj + delta, 1, resolution - 2)
                agent = proposed
    return SyntheticDataset(
        name="minipush",
        observations=observations,
        actions=actions,
        states=states,
        masks=masks,
        metadata={
            "env": "MiniPush",
            "seed": seed,
            "resolution": resolution,
            "state_names": ["agent_x", "agent_y", "object_x", "object_y", "goal_x", "goal_y"],
            "action_names": ["dx", "dy"],
            "mask_names": ["goal", "object", "agent"],
        },
    )
