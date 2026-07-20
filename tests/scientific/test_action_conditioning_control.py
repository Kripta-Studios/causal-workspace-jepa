from __future__ import annotations

import unittest

from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA, evaluate_latent_mse


class ActionConditioningControlTests(unittest.TestCase):
    def test_shuffled_actions_damage_latent_prediction(self) -> None:
        dataset = generate_pointmass2d(trajectories=24, steps=24, seed=11)
        train_obs = dataset.observations[:18]
        train_actions = dataset.actions[:18]
        test_obs = dataset.observations[18:]
        test_actions = dataset.actions[18:]
        conditioned = TinyActionConditionedJEPA.fit(train_obs, train_actions, latent_dim=16, seed=11)
        shuffled = TinyActionConditionedJEPA.fit(
            train_obs,
            train_actions,
            latent_dim=16,
            seed=11,
            action_mode="shuffled_action",
        )
        self.assertLess(
            evaluate_latent_mse(conditioned, test_obs, test_actions),
            evaluate_latent_mse(shuffled, test_obs, test_actions),
        )


if __name__ == "__main__":
    unittest.main()
