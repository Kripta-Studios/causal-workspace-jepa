from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from causal_workspace_jepa.common.types import LatentState
from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.models.tiny_jepa import (
    TinyActionConditionedJEPA,
    evaluate_latent_mse,
    mean_latent_baseline_mse,
)
from causal_workspace_jepa.planning.cem import random_shooting_plan
from causal_workspace_jepa.planning.closed_loop import pointmass_rollout_cost


class TinyJepaSmokeTests(unittest.TestCase):
    def test_train_save_load_and_plan(self) -> None:
        dataset = generate_pointmass2d(trajectories=20, steps=20, seed=0)
        train_obs = dataset.observations[:16]
        train_actions = dataset.actions[:16]
        test_obs = dataset.observations[16:]
        test_actions = dataset.actions[16:]
        model = TinyActionConditionedJEPA.fit(train_obs, train_actions, latent_dim=12, seed=0)
        no_action = TinyActionConditionedJEPA.fit(
            train_obs,
            train_actions,
            latent_dim=12,
            seed=0,
            action_mode="no_action",
        )
        conditioned_mse = evaluate_latent_mse(model, test_obs, test_actions)
        self.assertLess(conditioned_mse, mean_latent_baseline_mse(model, test_obs))
        self.assertLess(conditioned_mse, evaluate_latent_mse(no_action, test_obs, test_actions))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.npz"
            model.save(path)
            loaded = TinyActionConditionedJEPA.load(path)
            self.assertAlmostEqual(conditioned_mse, evaluate_latent_mse(loaded, test_obs, test_actions))
        latent = model.encode(test_obs[0, 0])
        output = model.predict(LatentState(latent.tensor), test_actions[:1, :3], return_intermediates=True)
        self.assertIn("predictor.input", output.intermediates)
        self.assertEqual(output.predicted_latents.shape[1], 3)

    def test_random_shooting_beats_average_random_cost(self) -> None:
        dataset = generate_pointmass2d(trajectories=20, steps=24, seed=2)
        model = TinyActionConditionedJEPA.fit(
            dataset.observations[:16],
            dataset.actions[:16],
            latent_dim=16,
            seed=2,
        )
        start = dataset.observations[16, 0]
        goal = np.array([0.75, 0.75], dtype=np.float32)
        plan = random_shooting_plan(model, start, goal, horizon=8, candidates=128, seed=2)
        planned = pointmass_rollout_cost(start, plan["actions"], goal)  # type: ignore[arg-type]
        rng = np.random.default_rng(101)
        random_actions = rng.uniform(-1.0, 1.0, size=(128, 8, 2)).astype(np.float32)
        random_mean = np.mean([pointmass_rollout_cost(start, actions, goal) for actions in random_actions])
        self.assertLess(planned, random_mean)


if __name__ == "__main__":
    unittest.main()
