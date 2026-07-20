from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.gpt2_medium_intervention_smoke import (
    _intervention_features,
    _intervention_specs,
)
from causal_workspace_jepa.experiments.world_model.tier0_mechanistic_study import _recovery
from causal_workspace_jepa.experiments.world_model.tier0_mechanistic_study import (
    _patch_action_coordinates,
)
from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA


class Milestone3HelperTests(unittest.TestCase):
    def test_recovery_scores_target_patch(self) -> None:
        clean = np.array([0.0, 0.0])
        target = np.array([1.0, 0.0])
        self.assertAlmostEqual(_recovery(clean, target, target), 1.0)
        self.assertAlmostEqual(_recovery(clean, target, clean), 0.0)

    def test_gpt2_intervention_feature_encoding(self) -> None:
        specs = _intervention_specs(hidden_size=16, magnitude=0.5, feature_count=2)
        self.assertEqual(specs, [(0, -0.5), (0, 0.5), (4, -0.5), (4, 0.5)])
        features = _intervention_features(feature_id=4, magnitude=-0.5, hidden_size=16)
        self.assertEqual(features.shape, (17,))
        self.assertEqual(features[4], -0.5)
        self.assertEqual(features[-1], -0.5)

    def test_action_patch_replays_only_predictor_action_coordinates(self) -> None:
        dataset = generate_pointmass2d(trajectories=8, steps=8, seed=3)
        model = TinyActionConditionedJEPA.fit(
            dataset.observations,
            dataset.actions,
            latent_dim=8,
            seed=3,
        )
        latent = model.encode(dataset.observations[0, 0]).tensor[None, :]
        recipient_action = dataset.actions[0, 0][None, :]
        donor_action = dataset.actions[1, 0][None, :]
        prediction, patched_input = _patch_action_coordinates(
            model,
            latent,
            recipient_action,
            donor_action,
        )
        expected = np.concatenate(
            [latent, donor_action, np.ones((1, 1), dtype=np.float32)],
            axis=-1,
        )
        np.testing.assert_allclose(patched_input, expected)
        np.testing.assert_allclose(prediction, expected @ model.predictor)


if __name__ == "__main__":
    unittest.main()
