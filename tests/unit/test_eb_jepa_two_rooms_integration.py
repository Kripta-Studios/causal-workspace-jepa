from __future__ import annotations

import unittest

from causal_workspace_jepa.experiments.world_model.eb_jepa_planner_constraint import (
    evaluate_planner_constraint,
)
from causal_workspace_jepa.experiments.world_model.eb_jepa_two_rooms_integration import (
    evaluate_two_rooms_smoke,
)


REVISION = "966e61e9285b3a876f49b9774e9720d9a99a7925"


class EBJEPATwoRoomsIntegrationTests(unittest.TestCase):
    def test_complete_integration_payload_passes(self) -> None:
        payload = {
            "source": {"revision": REVISION, "clean": True},
            "runtime": {
                "python": "3.12.13",
                "torch": "2.10.0+cu128",
                "cuda_runtime": "12.8",
                "capability": [12, 0],
                "arch_list": ["sm_90", "sm_120"],
            },
            "dataset": {
                "batch_shapes": [
                    [2, 2, 9, 65, 65],
                    [2, 2, 9],
                    [2, 2, 9],
                    [2, 1],
                    [2, 1],
                ]
            },
            "training": {
                "loss_finite": True,
                "gradients_finite": True,
                "parameter_delta": 1e-3,
                "prediction_shape": [2, 512, 5, 1, 1],
                "encoder_parameters": 1_426_096,
                "predictor_parameters": 793_600,
            },
            "checkpoint": {"max_restore_error": 0.0, "bytes": 10},
            "planner": {"name": "MPPIPlanner", "action_shape": [1, 2], "finite": True},
            "memory": {"peak_reserved_bytes": 1_000_000},
        }
        self.assertTrue(all(evaluate_two_rooms_smoke(payload, REVISION).values()))

    def test_nonfinite_planner_is_rejected(self) -> None:
        payload = {
            "source": {"revision": REVISION, "clean": True},
            "runtime": {},
            "dataset": {},
            "training": {},
            "checkpoint": {},
            "planner": {"name": "MPPIPlanner", "action_shape": [1, 2], "finite": False},
            "memory": {},
        }
        self.assertFalse(evaluate_two_rooms_smoke(payload, REVISION)["integrated_mppi_plan_finite"])

    def test_frozen_planner_asymmetry_passes(self) -> None:
        payload = {
            "design": {"num_seeds": 32, "seeds": list(range(32)), "max_norm": 2.45},
            "source_contract": {
                "cem_plan_mentions_max_norms": True,
                "mppi_plan_mentions_max_norms": False,
                "environment_step_checks_action_space": False,
                "environment_step_passes_action_to_transition": True,
            },
            "summary": {
                "cem_violation_count": 0,
                "cem_max_observed_norm": 2.4,
                "mppi_violation_count": 32,
                "mppi_violation_fraction": 1.0,
                "mppi_median_max_norm": 6.4,
            },
        }
        self.assertTrue(all(evaluate_planner_constraint(payload).values()))

    def test_mppi_nonreplication_is_rejected(self) -> None:
        payload = {
            "design": {"num_seeds": 32, "seeds": list(range(32)), "max_norm": 2.45},
            "source_contract": {
                "cem_plan_mentions_max_norms": True,
                "mppi_plan_mentions_max_norms": False,
                "environment_step_checks_action_space": False,
                "environment_step_passes_action_to_transition": True,
            },
            "summary": {
                "cem_violation_count": 0,
                "cem_max_observed_norm": 2.4,
                "mppi_violation_count": 1,
                "mppi_violation_fraction": 1 / 32,
                "mppi_median_max_norm": 2.4,
            },
        }
        self.assertFalse(evaluate_planner_constraint(payload)["mppi_violation_replication"])


if __name__ == "__main__":
    unittest.main()
