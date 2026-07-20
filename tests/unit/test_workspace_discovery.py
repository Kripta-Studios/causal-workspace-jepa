from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.interpretability.workspace_tests import (
    discover_shared_subspace,
    project_out_subspace,
    random_subspace,
)


class WorkspaceDiscoveryTests(unittest.TestCase):
    def test_detector_finds_known_shared_subspace(self) -> None:
        rng = np.random.default_rng(4)
        basis = random_subspace(12, 3, seed=4)
        jacobians = {
            f"consumer_{index}": rng.normal(size=(4, 3)) @ basis.T for index in range(5)
        }
        result = discover_shared_subspace(
            jacobians,
            max_dimension=4,
            min_consumer_capture=0.99,
            max_compactness_ratio=0.4,
            min_consumers=5,
        )
        self.assertTrue(result.sensitivity_candidate_found)
        self.assertEqual(result.dimension, 3)
        self.assertGreater(result.min_capture, 0.99)

    def test_detector_rejects_disjoint_consumers(self) -> None:
        rng = np.random.default_rng(5)
        jacobians = {}
        for index in range(5):
            jacobian = np.zeros((3, 12), dtype=np.float64)
            jacobian[:, index * 2 : index * 2 + 2] = rng.normal(size=(3, 2))
            jacobians[f"consumer_{index}"] = jacobian
        result = discover_shared_subspace(
            jacobians,
            max_dimension=4,
            min_consumer_capture=0.75,
            max_compactness_ratio=0.4,
            min_consumers=5,
        )
        self.assertFalse(result.sensitivity_candidate_found)
        self.assertLess(result.min_capture, 0.75)

    def test_projection_removes_only_candidate_coordinates(self) -> None:
        activation = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
        basis = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)
        projected = project_out_subspace(activation, basis)
        np.testing.assert_allclose(projected, [[0.0, 3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
