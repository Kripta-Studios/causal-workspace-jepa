from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.world_model.lewm_population_geometry_study import (
    _action_pairs,
    _gauge_audit,
    _transport_scores,
)


class LeWMPopulationGeometryTests(unittest.TestCase):
    def test_action_pairs_cover_all_ordered_replacements(self) -> None:
        pairs = _action_pairs()
        self.assertEqual(len(pairs), 12)
        self.assertEqual(len(set(pairs)), 12)
        self.assertTrue(all(recipient != donor for recipient, donor in pairs))

    def test_transport_scores_are_exact_for_direct_replay(self) -> None:
        observed = np.asarray([[1.0, -2.0], [-3.0, 4.0]])
        scores = _transport_scores(observed.copy(), observed)
        self.assertEqual(scores["normalized_mse"], 0.0)
        self.assertAlmostEqual(scores["correlation"], 1.0)
        self.assertAlmostEqual(scores["mean_cosine"], 1.0)

    def test_decoded_endpoint_is_gauge_invariant_and_raw_ranking_is_not(self) -> None:
        rng = np.random.default_rng(23)
        observed = rng.normal(size=(16, 5))
        local = observed + rng.normal(scale=0.2, size=observed.shape)
        population = observed + rng.normal(scale=0.1, size=observed.shape)
        decoder = rng.normal(size=(5, 3))
        audit = _gauge_audit(local, population, observed, decoder, seed=29)
        self.assertLessEqual(audit["decoded_invariance_max_abs"], 1e-12)
        self.assertTrue(audit["analytic_ranking_flip"])


if __name__ == "__main__":
    unittest.main()
