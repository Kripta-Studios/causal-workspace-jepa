from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_element_layer_geometry_study import (
    ELEMENT_PAIRS,
    ELEMENT_SPLIT_IDS,
    _inversion_decision,
    _scores,
    _transition_decision,
    _within_split_pairs,
)


class QwenElementLayerGeometryTests(unittest.TestCase):
    def test_entity_roster_and_pairs_are_split_safe(self) -> None:
        self.assertEqual(len(ELEMENT_PAIRS), 36)
        self.assertEqual(len(set(ELEMENT_PAIRS)), 36)
        self.assertEqual(np.bincount(ELEMENT_SPLIT_IDS).tolist(), [24, 6, 6])
        pairs = _within_split_pairs(ELEMENT_SPLIT_IDS)
        self.assertEqual(len(pairs), 612)
        self.assertTrue(all(ELEMENT_SPLIT_IDS[a] == ELEMENT_SPLIT_IDS[b] for a, b in pairs))

    def test_scores_separate_vector_candidate_and_donor_endpoints(self) -> None:
        prediction = np.asarray([[2.0, -1.0], [-1.0, 2.0]])
        arrays = {
            "clean_answer_logits": np.zeros((2, 2)),
            "recipient_id": np.asarray([0, 1]),
            "donor_id": np.asarray([0, 1]),
            "intervened_answer_logits": prediction[None],
        }
        score = _scores(
            prediction,
            prediction,
            arrays,
            0,
            np.ones(2, dtype=bool),
        )
        self.assertEqual(score["contrast_normalized_mse"], 0.0)
        self.assertAlmostEqual(score["contrast_correlation"], 1.0)
        self.assertEqual(score["answer_candidate_agreement"], 1.0)
        self.assertEqual(score["predicted_donor_candidate_rate"], 1.0)

    def test_registered_transition_and_inversion_gates_require_both_splits(self) -> None:
        behavior = {
            str(layer): {
                split: {
                    "full_vocab_donor_token_transfer": value,
                }
                for split in ("validation", "test")
            }
            for layer, value in ((18, 0.0), (21, 0.1), (24, 0.8), (26, 0.9))
        }
        self.assertTrue(_transition_decision({}, behavior))
        scores = {
            str(layer): {
                "splits": {
                    split: {
                        "local_jacobian": {
                            "contrast_normalized_mse": local,
                            "contrast_correlation": local_corr,
                        },
                        "train_population_jacobian": {
                            "contrast_normalized_mse": population,
                            "contrast_correlation": population_corr,
                        },
                    }
                    for split in ("validation", "test")
                }
            }
            for layer, local, population, local_corr, population_corr in (
                (21, 0.02, 0.20, 0.99, 0.90),
                (24, 0.30, 0.10, 0.94, 0.97),
            )
        }
        self.assertTrue(_inversion_decision({}, scores))
        behavior["24"]["test"]["full_vocab_donor_token_transfer"] = 0.2
        self.assertFalse(_transition_decision({}, behavior))


if __name__ == "__main__":
    unittest.main()
