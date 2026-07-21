from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_target_encoder_ijepa_study import (
    _PCAProjector,
    _combined_scores,
)


class QwenTargetEncoderStudyTests(unittest.TestCase):
    def test_behavior_score_uses_direct_answer_candidate(self) -> None:
        observed = np.asarray([[1.0, -1.0, 2.0, -2.0], [-1.0, 1.0, -2.0, 2.0]])
        data = {
            "clean_answer_logits": np.zeros((2, 2), dtype=np.float32),
            "intervened_answer_logits": np.asarray([[2.0, -2.0], [-2.0, 2.0]]),
            "donor_id": np.asarray([0, 1]),
        }
        mask = np.ones(2, dtype=bool)
        score = _combined_scores(observed.copy(), observed, data, mask)
        self.assertEqual(score["normalized_mse"], 0.0)
        self.assertEqual(score["answer_candidate_agreement"], 1.0)
        self.assertEqual(score["predicted_donor_candidate_rate"], 1.0)

    def test_pca_projector_fits_training_coordinates_only(self) -> None:
        rng = np.random.default_rng(23)
        train = rng.normal(size=(20, 8)).astype(np.float32)
        projector = _PCAProjector.fit(train, components=3)
        transformed = projector.transform(train[:4])
        self.assertEqual(transformed.shape, (4, 3))
        self.assertTrue(np.isfinite(transformed).all())


if __name__ == "__main__":
    unittest.main()
