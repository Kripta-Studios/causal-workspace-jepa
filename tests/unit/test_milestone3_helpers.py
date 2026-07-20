from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.gpt2_medium_intervention_smoke import (
    _intervention_features,
    _intervention_specs,
)
from causal_workspace_jepa.experiments.world_model.tier0_mechanistic_study import _recovery


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


if __name__ == "__main__":
    unittest.main()
