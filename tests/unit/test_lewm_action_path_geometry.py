from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.world_model.lewm_action_path_geometry_study import (
    _composite_legendre,
    _spearman,
    _stratified_permutation_null,
    _stratified_profile_indices,
)


class LeWMActionPathGeometryTests(unittest.TestCase):
    def test_composite_legendre_integrates_polynomial(self) -> None:
        nodes, weights = _composite_legendre(order=4, panels=5)
        self.assertEqual(len(nodes), 20)
        self.assertTrue(np.all(np.diff(nodes) > 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(float(np.sum(weights * nodes**6)), 1.0 / 7.0, places=12)

    def test_profile_sampling_is_action_pair_stratified(self) -> None:
        contexts, pairs = _stratified_profile_indices(
            20, contexts_per_pair=3, seed=11
        )
        self.assertEqual(len(contexts), 36)
        self.assertEqual(np.bincount(pairs).tolist(), [3] * 12)
        for pair_id in range(12):
            self.assertEqual(len(set(contexts[pairs == pair_id])), 3)

    def test_stratified_null_breaks_within_pair_association(self) -> None:
        predictor = np.tile(np.arange(4, dtype=np.float64), 12)
        target = predictor.copy()
        strata = np.repeat(np.arange(12), 4)
        null = _stratified_permutation_null(
            predictor, target, strata, permutations=128, seed=13
        )
        self.assertAlmostEqual(_spearman(predictor, target), 1.0)
        self.assertLess(null["p95_spearman"], 0.99)


if __name__ == "__main__":
    unittest.main()
