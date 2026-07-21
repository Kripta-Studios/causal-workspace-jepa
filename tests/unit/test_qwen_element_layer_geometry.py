from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_element_layer_geometry_study import (
    COUNTRY_CODE_PAIRS,
    COUNTRY_CODE_SPLIT_IDS,
    ELEMENT_PAIRS,
    ELEMENT_SPLIT_IDS,
    STATE_PAIRS,
    STATE_SPLIT_IDS,
    _bounded_lag_decision,
    _boundary_alignment_decision,
    _boundary_alignment_details,
    _boundary_sign_equality_decision,
    _control_population_association_decision,
    _inversion_decision,
    _monotone_terminal_transition_decision,
    _scores,
    _terminal_transition_decision,
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
        self.assertEqual(len(STATE_PAIRS), 36)
        self.assertEqual(len(set(STATE_PAIRS)), 36)
        self.assertEqual(np.bincount(STATE_SPLIT_IDS).tolist(), [24, 6, 6])
        state_pairs = _within_split_pairs(STATE_SPLIT_IDS)
        self.assertEqual(len(state_pairs), 612)
        self.assertTrue(
            all(STATE_SPLIT_IDS[a] == STATE_SPLIT_IDS[b] for a, b in state_pairs)
        )
        self.assertEqual(len(COUNTRY_CODE_PAIRS), 36)
        self.assertEqual(len(set(COUNTRY_CODE_PAIRS)), 36)
        self.assertEqual(np.bincount(COUNTRY_CODE_SPLIT_IDS).tolist(), [24, 6, 6])
        country_pairs = _within_split_pairs(COUNTRY_CODE_SPLIT_IDS)
        self.assertEqual(len(country_pairs), 612)
        self.assertTrue(
            all(
                COUNTRY_CODE_SPLIT_IDS[a] == COUNTRY_CODE_SPLIT_IDS[b]
                for a, b in country_pairs
            )
        )
        self.assertTrue(
            set(answer for _entity, answer in COUNTRY_CODE_PAIRS).isdisjoint(
                answer for _entity, answer in STATE_PAIRS
            )
        )

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

    def test_control_conditioned_population_gate_requires_both_splits(self) -> None:
        details = {
            split: {
                "by_layer": {
                    "21": {"population_advantage": -0.10},
                    "24": {"population_advantage": 0.02},
                },
                "donor_control_advantage_spearman": 0.9,
            }
            for split in ("validation", "test")
        }
        self.assertTrue(_control_population_association_decision({}, details))
        details["test"]["by_layer"]["24"]["population_advantage"] = -0.01
        self.assertFalse(_control_population_association_decision({}, details))

    def test_boundary_alignment_uses_observed_onset_not_fixed_layer(self) -> None:
        details = {
            split: {
                "by_layer": {
                    "18": {
                        "population_advantage": -0.5,
                        "full_vocab_donor_transfer": 0.0,
                    },
                    "21": {
                        "population_advantage": -0.3,
                        "full_vocab_donor_transfer": 0.0,
                    },
                    "24": {
                        "population_advantage": -0.01,
                        "full_vocab_donor_transfer": 0.4,
                    },
                    "26": {
                        "population_advantage": 0.08,
                        "full_vocab_donor_transfer": 0.7,
                    },
                },
                "donor_control_advantage_spearman": 0.95,
            }
            for split in ("validation", "test")
        }
        boundaries = _boundary_alignment_details({}, details)
        self.assertEqual(boundaries["validation"]["control_boundary_layer"], 26)
        self.assertEqual(
            boundaries["validation"]["population_advantage_boundary_layer"], 26
        )
        self.assertTrue(_boundary_alignment_decision({}, details, boundaries))
        self.assertTrue(_boundary_sign_equality_decision({}, details, boundaries))
        details["test"]["by_layer"]["24"]["full_vocab_donor_transfer"] = 0.6
        boundaries = _boundary_alignment_details({}, details)
        self.assertFalse(_boundary_alignment_decision({}, details, boundaries))

    def test_terminal_transition_does_not_require_layer_24_control(self) -> None:
        behavior = {
            str(layer): {
                split: {"full_vocab_donor_token_transfer": value}
                for split in ("validation", "test")
            }
            for layer, value in ((18, 0.0), (21, 0.0), (24, 0.2), (26, 0.7))
        }
        self.assertTrue(_terminal_transition_decision({}, behavior))

    def test_bounded_lag_accepts_zero_or_one_later_grid_step(self) -> None:
        details = {
            split: {
                "by_layer": {
                    "18": {"population_advantage": -0.5},
                    "21": {"population_advantage": -0.2},
                    "24": {"population_advantage": -0.01},
                    "26": {"population_advantage": 0.08},
                },
                "donor_control_advantage_spearman": 0.9,
            }
            for split in ("validation", "test")
        }
        boundaries = {
            split: {
                "control_boundary_layer": 24,
                "population_advantage_boundary_layer": 26,
            }
            for split in ("validation", "test")
        }
        self.assertTrue(_bounded_lag_decision({}, details, boundaries))
        boundaries["test"]["population_advantage_boundary_layer"] = 21
        self.assertFalse(_bounded_lag_decision({}, details, boundaries))

    def test_monotone_terminal_transition_allows_layer_21_control(self) -> None:
        behavior = {
            str(layer): {
                split: {"full_vocab_donor_token_transfer": value}
                for split in ("validation", "test")
            }
            for layer, value in ((18, 0.0), (21, 0.23), (24, 0.95), (26, 1.0))
        }
        self.assertTrue(_monotone_terminal_transition_decision({}, behavior))
        behavior["24"]["test"]["full_vocab_donor_token_transfer"] = 0.10
        self.assertFalse(_monotone_terminal_transition_decision({}, behavior))


if __name__ == "__main__":
    unittest.main()
