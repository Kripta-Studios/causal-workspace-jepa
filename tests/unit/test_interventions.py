from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.hooks.interventions import (
    apply_intervention,
    matched_random_feature_control,
    spec_from_json,
    spec_to_json,
)


class InterventionTests(unittest.TestCase):
    def test_zero_patch_and_steer_positions_features(self) -> None:
        activation = np.ones((2, 3, 4), dtype=np.float32)
        zero = InterventionSpec(site="x", operation="zero", positions=(1,), feature_ids=(2,))
        changed = apply_intervention(activation, zero)
        self.assertEqual(changed[0, 1, 2], 0.0)
        self.assertEqual(changed[0, 0, 2], 1.0)
        donor = np.full_like(activation, 7.0)
        patch = InterventionSpec(site="x", operation="patch", positions=(2,), feature_ids=(1,))
        patched = apply_intervention(activation, patch, donor=donor)
        self.assertEqual(patched[1, 2, 1], 7.0)
        steer = InterventionSpec(site="x", operation="steer", positions=(0,), feature_ids=(3,), magnitude=2.5)
        steered = apply_intervention(activation, steer)
        self.assertEqual(steered[0, 0, 3], 3.5)

    def test_project_out_removes_basis_direction(self) -> None:
        activation = np.array([[3.0, 4.0]], dtype=np.float32)
        spec = InterventionSpec(site="x", operation="project_out")
        changed = apply_intervention(activation, spec, basis=np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(changed, np.array([[0.0, 4.0]], dtype=np.float32), atol=1e-6)

    def test_spec_json_and_matched_control(self) -> None:
        spec = InterventionSpec(site="x", operation="scale", feature_ids=(0, 1), magnitude=0.5, seed=1)
        self.assertEqual(spec_from_json(spec_to_json(spec)), spec)
        control = matched_random_feature_control(spec, hidden_size=8, seed=3)
        self.assertEqual(control.operation, spec.operation)
        self.assertEqual(len(control.feature_ids or ()), 2)


if __name__ == "__main__":
    unittest.main()
