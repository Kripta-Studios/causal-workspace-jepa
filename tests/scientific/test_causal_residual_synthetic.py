from __future__ import annotations

import unittest

from causal_workspace_jepa.experiments.cross_domain.causal_residual_synthetic import (
    run_stage0_benchmark,
)


class CausalResidualSyntheticStage0Tests(unittest.TestCase):
    def test_stage0_falsification_cases_and_determinism(self) -> None:
        first = run_stage0_benchmark(seed=20260802)
        second = run_stage0_benchmark(seed=20260802)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.status, "SMOKE_VALIDATED")
        self.assertEqual(
            [case.case_id for case in first.cases],
            [
                "linear_zero_residual",
                "quadratic_hvp_zero_false_discovery",
                "nonlinear_compositional_residual",
                "predictable_nuisance_guard",
            ],
        )

        linear = first.by_case("linear_zero_residual")
        self.assertEqual(linear.status, "NEGATIVE_RESULT")
        self.assertEqual(linear.metrics["selected_baseline"], "exact_jvp")
        self.assertLess(float(linear.metrics["baseline_replay_mse"]), 1e-20)
        self.assertLess(float(linear.metrics["max_abs_residual"]), 1e-10)
        self.assertFalse(bool(linear.metrics["learned_discovery_permitted"]))

        quadratic = first.by_case("quadratic_hvp_zero_false_discovery")
        self.assertEqual(quadratic.status, "NEGATIVE_RESULT")
        self.assertEqual(quadratic.metrics["selected_baseline"], "quadratic_hvp")
        self.assertLess(float(quadratic.metrics["baseline_replay_mse"]), 1e-20)
        self.assertLess(float(quadratic.metrics["max_abs_residual"]), 1e-10)
        self.assertFalse(bool(quadratic.metrics["learned_discovery_permitted"]))

        nonlinear = first.by_case("nonlinear_compositional_residual")
        self.assertEqual(nonlinear.status, "SMOKE_VALIDATED")
        self.assertGreater(float(nonlinear.metrics["eligible_baseline_replay_mse"]), 1e-4)
        self.assertLess(
            float(nonlinear.metrics["residual_learner_replay_mse"]),
            float(nonlinear.metrics["eligible_baseline_replay_mse"]) * 0.05,
        )
        self.assertGreater(float(nonlinear.metrics["direct_replay_improvement_fraction"]), 0.95)

        nuisance = first.by_case("predictable_nuisance_guard")
        self.assertEqual(nuisance.status, "SMOKE_VALIDATED")
        self.assertEqual(
            nuisance.metrics["observable_target_contract"],
            "physical_state_delta_and_frozen_known_nuisance_mask",
        )
        self.assertEqual(int(nuisance.metrics["naive_training_steps"]), 220)
        self.assertEqual(int(nuisance.metrics["guarded_training_steps"]), 220)
        self.assertGreater(float(nuisance.metrics["naive_nuisance_probe_accuracy"]), 0.60)
        self.assertGreater(float(nuisance.metrics["naive_low_dim_nuisance_dominance_r2"]), 0.55)
        self.assertGreater(float(nuisance.metrics["naive_variance_explained_episode_id"]), 0.8)
        self.assertGreater(float(nuisance.metrics["naive_variance_explained_camera_template"]), 0.8)
        self.assertLess(float(nuisance.metrics["naive_causal_effect_probe_r2"]), 0.2)
        self.assertEqual(float(nuisance.metrics["naive_conditional_action_sensitivity"]), 0.0)
        self.assertGreater(
            float(nuisance.metrics["naive_next_state_mse_after_nuisance_removal"]),
            float(nuisance.metrics["naive_next_state_mse"]) * 1.05,
        )
        self.assertLess(float(nuisance.metrics["guarded_residual_mse"]), 0.02)
        self.assertGreater(float(nuisance.metrics["guarded_causal_effect_probe_r2"]), 0.65)
        self.assertGreater(
            float(nuisance.metrics["guarded_causal_effect_probe_accuracy"]),
            float(nuisance.metrics["naive_causal_effect_probe_accuracy"]) + 0.15,
        )
        self.assertGreater(float(nuisance.metrics["guarded_norm_matched_action_sensitivity"]), 0.1)
        self.assertGreater(
            float(nuisance.metrics["guarded_covariance_matched_action_sensitivity"]), 0.1
        )
        self.assertLess(
            float(nuisance.metrics["guarded_residual_mse_after_nuisance_removal"]),
            float(nuisance.metrics["guarded_residual_mse"]) * 2.5 + 1e-4,
        )
        self.assertLess(float(nuisance.metrics["norm_matched_control_max_norm_difference"]), 1e-6)
        self.assertLess(
            float(nuisance.metrics["covariance_matched_control_max_norm_difference"]), 1e-6
        )
        self.assertLess(
            float(nuisance.metrics["covariance_matched_control_max_covariance_difference"]),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
