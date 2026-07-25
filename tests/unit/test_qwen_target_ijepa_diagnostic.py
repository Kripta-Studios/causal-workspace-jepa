from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_target_ijepa_diagnostic import (
    TrainOnlyPCA,
    compute_target_ijepa_diagnostic,
    effective_rank,
    eta_squared,
)


class _SyntheticLatentModel:
    def target_latent(self, values: np.ndarray) -> np.ndarray:
        return values[:, :2].astype(np.float32)

    def predict_latent(
        self,
        clean_source: np.ndarray,
        donor_source: np.ndarray,
        clean_target: np.ndarray,
        source_delta: np.ndarray,
    ) -> np.ndarray:
        del clean_source, donor_source
        return (clean_target[:, :2] + 0.5 * source_delta[:, :2]).astype(np.float32)


def _synthetic_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    examples = 24
    hidden = 4
    split = np.repeat(np.arange(3), 8)
    recipient = np.tile(np.repeat(np.arange(4), 2), 3)
    donor = np.tile(np.asarray([1, 2, 2, 3, 3, 0, 0, 1]), 3)
    clean_source = rng.normal(size=(examples, hidden)).astype(np.float32)
    donor_source = rng.normal(size=(examples, hidden)).astype(np.float32)
    source_delta = donor_source - clean_source
    clean_target = (0.3 * clean_source).astype(np.float32)
    donor_signal = np.eye(4, dtype=np.float32)[donor]
    intervened = clean_target + donor_signal
    hidden_effect = intervened - clean_target
    logit_effect = hidden_effect[:, :2]
    return {
        "split_id": split,
        "recipient_id": recipient,
        "donor_id": donor,
        "clean_source": clean_source,
        "donor_source": donor_source,
        "source_delta": source_delta,
        "clean_target_hidden": clean_target,
        "intervened_target_hidden": intervened,
        "clean_answer_logits": np.zeros((examples, 2), dtype=np.float32),
        "target_effect": np.concatenate([hidden_effect, logit_effect], axis=1),
    }


class QwenTargetIJEPADiagnosticTests(unittest.TestCase):
    def test_train_only_pca_ignores_heldout_rows_when_fitted(self) -> None:
        rng = np.random.default_rng(11)
        training = rng.normal(size=(20, 5)).astype(np.float32)
        heldout_a = rng.normal(size=(6, 5)).astype(np.float32)
        heldout_b = heldout_a + 1000.0
        first = TrainOnlyPCA.fit(training, 3)
        second = TrainOnlyPCA.fit(np.concatenate([training, heldout_a])[: len(training)], 3)
        third = TrainOnlyPCA.fit(np.concatenate([training, heldout_b])[: len(training)], 3)
        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(second.mean, third.mean)
        np.testing.assert_array_equal(first.components, second.components)
        np.testing.assert_array_equal(second.components, third.components)

    def test_oracle_training_scores_do_not_depend_on_heldout_values(self) -> None:
        original = _synthetic_data()
        shifted = {name: values.copy() for name, values in original.items()}
        heldout = shifted["split_id"] != 0
        for name in ("clean_target_hidden", "intervened_target_hidden", "target_effect"):
            shifted[name][heldout] += 1000.0
        first = compute_target_ijepa_diagnostic(
            original, {5: _SyntheticLatentModel()}, pca_dimensions=(1, 2)
        )
        second = compute_target_ijepa_diagnostic(
            shifted, {5: _SyntheticLatentModel()}, pca_dimensions=(1, 2)
        )
        for dimension in ("1", "2"):
            self.assertEqual(
                first["oracle_pca_ridge"][dimension]["state_pair"]["train"],
                second["oracle_pca_ridge"][dimension]["state_pair"]["train"],
            )
            self.assertEqual(
                first["oracle_pca_ridge"][dimension]["causal_delta"]["train"],
                second["oracle_pca_ridge"][dimension]["causal_delta"]["train"],
            )

    def test_eta_squared_and_effective_rank_have_known_limits(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        grouped = np.asarray([[0.0, 0.0], [0.0, 0.0], [2.0, 1.0], [2.0, 1.0]])
        self.assertAlmostEqual(eta_squared(grouped, labels), 1.0)
        self.assertAlmostEqual(effective_rank(np.asarray([[0.0, 0.0], [1.0, 0.0]])), 1.0)

    def test_synthetic_diagnostic_reports_all_splits_and_posthoc_oracles(self) -> None:
        data = _synthetic_data()
        first = compute_target_ijepa_diagnostic(
            data, {5: _SyntheticLatentModel()}, pca_dimensions=(1, 2), decoder_ridge=1.0
        )
        second = compute_target_ijepa_diagnostic(
            data, {5: _SyntheticLatentModel()}, pca_dimensions=(1, 2), decoder_ridge=1.0
        )
        self.assertEqual(first, second)
        self.assertEqual(first["split_counts"], {"train": 8, "validation": 8, "test": 8})
        self.assertEqual(first["pca_fit_split"], "train")
        self.assertEqual(
            set(first["effective_rank_by_seed_and_split"]["5"]),
            {"train", "validation", "test"},
        )
        self.assertGreater(
            first["raw_identity_eta_squared_by_split"]["train"][
                "intervened_target_donor_eta_squared"
            ],
            0.5,
        )
        self.assertGreater(
            first["raw_identity_eta_squared_by_split"]["train"][
                "clean_target_recipient_eta_squared"
            ],
            first["raw_identity_eta_squared_by_split"]["train"][
                "clean_target_donor_eta_squared"
            ],
        )
        for dimension in ("1", "2"):
            self.assertEqual(
                set(first["oracle_pca_ridge"][dimension]["state_pair"]),
                set(("train", "validation", "test")),
            )
            self.assertTrue(
                np.isfinite(
                    first["oracle_pca_ridge"][dimension]["causal_delta"]["test"][
                        "normalized_mse"
                    ]
                )
            )

    def test_diagnostic_rejects_nonfinite_inputs_and_empty_models(self) -> None:
        data = _synthetic_data()
        with self.assertRaisesRegex(ValueError, "at least one"):
            compute_target_ijepa_diagnostic(data, {}, pca_dimensions=(1, 2))
        data["target_effect"][0, 0] = np.nan
        with self.assertRaisesRegex(FloatingPointError, "target_effect"):
            compute_target_ijepa_diagnostic(
                data, {5: _SyntheticLatentModel()}, pca_dimensions=(1, 2)
            )


if __name__ == "__main__":
    unittest.main()
