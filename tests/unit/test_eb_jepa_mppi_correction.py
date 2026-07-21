from __future__ import annotations

import unittest

import torch

from causal_workspace_jepa.experiments.world_model.eb_jepa_mppi_correction import (
    evaluate_mppi_correction,
)
from causal_workspace_jepa.planning.eb_jepa_constrained_mppi import project_action_norms


class EBJEPAMPPIConstraintCorrectionTests(unittest.TestCase):
    def test_projection_enforces_selected_norm(self) -> None:
        actions = torch.tensor([[[3.0, 4.0], [0.3, 0.4]]])
        projected = project_action_norms(
            actions,
            max_norms=[2.45],
            max_norm_dims=[0, 1],
        )
        self.assertLessEqual(float(projected.norm(dim=-1).max()), 2.450001)
        self.assertTrue(torch.allclose(projected[0, 1], actions[0, 1]))

    def test_no_bound_is_exact_identity(self) -> None:
        actions = torch.randn(3, 4, 2)
        projected = project_action_norms(actions, max_norms=None, max_norm_dims=None)
        self.assertIs(projected, actions)

    def test_registered_correction_payload_passes(self) -> None:
        payload = {
            "design": {"num_seeds": 32, "seeds": list(range(32)), "max_norm": 2.45},
            "summary": {
                "max_unbounded_action_abs_diff": 0.0,
                "max_unbounded_loss_abs_diff": 0.0,
                "corrected_returned_violation_count": 0,
                "corrected_cost_input_violation_count": 0,
                "corrected_max_returned_norm": 2.449999,
                "corrected_max_cost_input_norm": 2.449999,
            },
        }
        self.assertTrue(all(evaluate_mppi_correction(payload).values()))

    def test_cost_input_violation_fails(self) -> None:
        payload = {
            "design": {"num_seeds": 32, "seeds": list(range(32)), "max_norm": 2.45},
            "summary": {
                "max_unbounded_action_abs_diff": 0.0,
                "max_unbounded_loss_abs_diff": 0.0,
                "corrected_returned_violation_count": 0,
                "corrected_cost_input_violation_count": 1,
                "corrected_max_returned_norm": 2.4,
                "corrected_max_cost_input_norm": 3.0,
            },
        }
        self.assertFalse(evaluate_mppi_correction(payload)["cost_inputs_are_bounded"])


if __name__ == "__main__":
    unittest.main()
