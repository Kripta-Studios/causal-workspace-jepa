from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.data.synthetic.bouncing_ball import generate_bouncing_ball2d
from causal_workspace_jepa.data.synthetic.minipush import generate_minipush
from causal_workspace_jepa.data.synthetic.multitask_pointmass import (
    generate_multitask_pointmass2d,
)
from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.data.synthetic.tiny_maze import generate_tiny_maze, step_tiny_maze
from causal_workspace_jepa.data.synthetic.two_body import generate_two_body_collision
from causal_workspace_jepa.experiments.world_model.multitask_workspace_study import (
    _combination_ids,
    _task_splits,
)


class SyntheticGeneratorTests(unittest.TestCase):
    def test_pointmass_is_deterministic(self) -> None:
        first = generate_pointmass2d(trajectories=3, steps=5, seed=7)
        second = generate_pointmass2d(trajectories=3, steps=5, seed=7)
        np.testing.assert_allclose(first.observations, second.observations)
        np.testing.assert_allclose(first.actions, second.actions)

    def test_all_tier0_shapes(self) -> None:
        generators = [
            generate_pointmass2d,
            generate_bouncing_ball2d,
            generate_two_body_collision,
            generate_tiny_maze,
            generate_minipush,
        ]
        for generator in generators:
            dataset = generator(trajectories=2, steps=4, seed=1)
            self.assertEqual(dataset.observations.shape[0], 2)
            self.assertEqual(dataset.observations.shape[1], 4)
            self.assertEqual(dataset.actions.shape[1], 3)
            self.assertEqual(dataset.transitions, 6)

    def test_multitask_pointmass_balances_context_and_uses_action(self) -> None:
        first = generate_multitask_pointmass2d(trajectories=16, steps=5, seed=9)
        second = generate_multitask_pointmass2d(trajectories=16, steps=5, seed=9)
        np.testing.assert_allclose(first.observations, second.observations)
        self.assertEqual(first.observations.shape, (16, 5, 8))
        contexts = np.unique(first.observations[:, 0, 4:], axis=0)
        self.assertEqual(len(contexts), 8)
        np.testing.assert_allclose(first.observations[:, :, 6:].sum(axis=-1), 1.0)

    def test_multitask_split_holds_out_one_composition(self) -> None:
        dataset = generate_multitask_pointmass2d(trajectories=80, steps=3, seed=13)
        combo_ids = _combination_ids(dataset.observations[:, 0])
        splits = _task_splits(combo_ids, heldout_combo=7, seed=13)
        np.testing.assert_array_equal(combo_ids[splits["test"]], 7)
        for name in ("predictor_train", "calibration", "consumer_train"):
            self.assertNotIn(7, combo_ids[splits[name]])
        combined = np.concatenate(list(splits.values()))
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(combo_ids)))

    def test_tiny_maze_wall_blocks_motion(self) -> None:
        position = np.array([1, 2])
        # Action 3 moves right into the central wall at (2, 2).
        next_position = step_tiny_maze(position, 3)
        np.testing.assert_array_equal(next_position, position)


if __name__ == "__main__":
    unittest.main()
