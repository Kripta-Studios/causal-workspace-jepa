from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_population_jacobian_confirmation import (
    _paired_bootstrap,
    _scores,
)


class PopulationJacobianConfirmationTests(unittest.TestCase):
    def test_scores_keep_continuous_and_candidate_endpoints_separate(self) -> None:
        observed = np.asarray([[2.0, -2.0], [-2.0, 2.0]])
        prediction = observed.copy()
        patch = {
            "clean_answer_logits": np.zeros((2, 2)),
            "intervened_answer_logits": observed.copy(),
            "donor_id": np.asarray([0, 1]),
        }
        score = _scores(prediction, observed, patch, np.ones(2, dtype=bool))
        self.assertEqual(score["normalized_mse"], 0.0)
        self.assertEqual(score["answer_candidate_agreement"], 1.0)
        self.assertEqual(score["predicted_donor_candidate_rate"], 1.0)

    def test_paired_bootstrap_detects_uniform_improvement(self) -> None:
        observed = np.ones((8, 3))
        local = np.zeros_like(observed)
        population = np.full_like(observed, 0.9)
        result = _paired_bootstrap(
            local,
            population,
            observed,
            draws=200,
            rng=np.random.default_rng(13),
        )
        self.assertGreater(result["ci95_low"], 0.0)
        self.assertEqual(result["probability_improvement_positive"], 1.0)


if __name__ == "__main__":
    unittest.main()
