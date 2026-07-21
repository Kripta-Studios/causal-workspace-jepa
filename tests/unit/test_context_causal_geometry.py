from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.interpretability.context_causal_geometry import (
    analytic_pooling_counterexample,
    causal_coupling,
    coupling_spectrum,
    euclidean_subspace_overlap,
    pooled_euclidean_overlap,
    transform_vector_covector_pair,
)


class ContextCausalGeometryTests(unittest.TestCase):
    def test_analytic_pooling_counterexample(self) -> None:
        result = analytic_pooling_counterexample()
        self.assertTrue(result["passed"])
        self.assertEqual(result["matched_max_abs_coupling"], 0.0)
        self.assertEqual(result["pooled_overlap"], 1.0)

    def test_coupling_is_invariant_to_general_coordinate_change(self) -> None:
        rng = np.random.default_rng(29)
        directions = rng.normal(size=(5, 7))
        jacobian = rng.normal(size=(3, 7))
        transform = rng.normal(size=(7, 7)) + 3.0 * np.eye(7)
        transformed_directions, transformed_jacobian = transform_vector_covector_pair(
            directions, jacobian, transform
        )
        np.testing.assert_allclose(
            causal_coupling(jacobian, directions),
            causal_coupling(transformed_jacobian, transformed_directions),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_naive_pooled_overlap_discards_pairing(self) -> None:
        directions = [np.eye(4)[:2], np.eye(4)[2:]]
        jacobians = [np.eye(4)[2:], np.eye(4)[:2]]
        original = pooled_euclidean_overlap(directions, jacobians, rank=4)
        permuted = pooled_euclidean_overlap(directions, jacobians[::-1], rank=4)
        self.assertAlmostEqual(original, permuted, places=12)
        self.assertLess(euclidean_subspace_overlap(directions[0], jacobians[0], rank=2), 1e-12)

    def test_coupling_spectrum_reports_rank(self) -> None:
        result = coupling_spectrum(np.diag([3.0, 1.0, 0.0]))
        self.assertGreater(result["effective_rank"], 1.0)
        self.assertLess(result["effective_rank"], 2.0)
        self.assertAlmostEqual(result["top_mode_energy_fraction"], 0.9)


if __name__ == "__main__":
    unittest.main()
