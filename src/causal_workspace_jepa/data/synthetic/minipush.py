"""MiniPush 32x32 pixel smoke generator with masks."""

from __future__ import annotations

import numpy as np

from causal_workspace_jepa.data.synthetic.base import SyntheticDataset


def _draw_square(image: np.ndarray, mask: np.ndarray, x: int, y: int, value: int) -> None:
    image[max(0, y - 1) : min(image.shape[0], y + 2), max(0, x - 1) : min(image.shape[1], x + 2)] = value
    mask[max(0, y - 1) : min(mask.shape[0], y + 2), max(0, x - 1) : min(mask.shape[1], x + 2)] = 1


CARDINALS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)


def quantize_minipush_actions(actions: np.ndarray) -> np.ndarray:
    """Map continuous actions to the nearest MiniPush cardinal."""

    packed = np.asarray(actions, dtype=np.float32)
    original_shape = packed.shape
    flat = packed.reshape(-1, original_shape[-1])
    distances = np.sum((flat[:, None, :] - CARDINALS[None, :, :].astype(np.float32)) ** 2, axis=-1)
    nearest = CARDINALS[np.argmin(distances, axis=1)]
    return nearest.reshape(original_shape).astype(np.float32)


def step_minipush(state: np.ndarray, action: np.ndarray, *, resolution: int = 32) -> np.ndarray:
    """One MiniPush step. Matches ``generate_minipush`` contact and clipping."""

    agent = np.asarray(state[:2], dtype=np.int64)
    obj = np.asarray(state[2:4], dtype=np.int64)
    goal = np.asarray(state[4:6], dtype=np.float32)
    delta = quantize_minipush_actions(np.asarray(action, dtype=np.float32)).astype(np.int64)
    proposed = np.clip(agent + delta, 1, resolution - 2)
    if np.max(np.abs(proposed - obj)) <= 1:
        obj = np.clip(obj + delta, 1, resolution - 2)
    agent = proposed
    return np.array([*agent.tolist(), *obj.tolist(), *goal.tolist()], dtype=np.float32)


def minipush_rollout_state(
    initial_state: np.ndarray,
    actions: np.ndarray,
    *,
    resolution: int = 32,
) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float32)
    for action in actions:
        state = step_minipush(state, action, resolution=resolution)
    return state


def constructed_goal_observation(start_state: np.ndarray) -> np.ndarray:
    """Identical goal vector for every planner: agent stays, object placed at goal xy."""

    start = np.asarray(start_state, dtype=np.float32)
    return np.array([start[0], start[1], start[4], start[5], start[4], start[5]], dtype=np.float32)


def object_goal_l2(state: np.ndarray) -> float:
    state = np.asarray(state, dtype=np.float32)
    return float(np.linalg.norm(state[2:4] - state[4:6]))


def manhattan_xy(left: np.ndarray, right: np.ndarray) -> int:
    left = np.asarray(left)
    right = np.asarray(right)
    return int(abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1])))


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
