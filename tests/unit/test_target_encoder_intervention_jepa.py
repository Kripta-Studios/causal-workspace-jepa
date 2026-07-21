from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from causal_workspace_jepa.models.target_encoder_intervention_jepa import (
    StandardizedRidgeDecoder,
    TargetEncoderInterventionJEPA,
)


class TargetEncoderInterventionJEPATests(unittest.TestCase):
    def test_fit_checkpoint_and_posthoc_decoder(self) -> None:
        rng = np.random.default_rng(17)
        recipient = rng.normal(size=(64, 12)).astype(np.float32)
        donor = rng.normal(size=(64, 12)).astype(np.float32)
        clean_target = (0.4 * recipient + 0.1).astype(np.float32)
        delta = donor - recipient
        intervened = (clean_target + 0.3 * np.tanh(delta)).astype(np.float32)
        train = np.arange(48)
        validation = np.arange(48, 64)
        model = TargetEncoderInterventionJEPA.fit(
            recipient[train],
            donor[train],
            clean_target[train],
            intervened[train],
            delta[train],
            (
                recipient[validation],
                donor[validation],
                clean_target[validation],
                intervened[validation],
                delta[validation],
            ),
            hidden_dim=24,
            meta_dim=6,
            steps=30,
            validation_interval=5,
            patience=30,
            seed=19,
        )
        predicted = model.predict_latent(
            recipient[validation],
            donor[validation],
            clean_target[validation],
            delta[validation],
        )
        target = model.target_latent(intervened[validation])
        self.assertEqual(predicted.shape, (16, 6))
        self.assertEqual(target.shape, (16, 6))
        self.assertTrue(np.isfinite(predicted).all())
        collapse = model.collapse_metrics(
            recipient[validation],
            donor[validation],
            clean_target[validation],
            intervened[validation],
            delta[validation],
        )
        self.assertGreater(collapse["target_effective_rank"], 1.0)
        decoder = StandardizedRidgeDecoder.fit(
            model.target_latent(intervened[train]), intervened[train], ridge=1.0
        )
        self.assertEqual(decoder.predict(predicted).shape, intervened[validation].shape)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.npz"
            decoder_path = Path(directory) / "decoder.npz"
            model.save(model_path)
            decoder.save(decoder_path)
            reloaded = TargetEncoderInterventionJEPA.load(model_path)
            reloaded_decoder = StandardizedRidgeDecoder.load(decoder_path)
            np.testing.assert_array_equal(
                predicted,
                reloaded.predict_latent(
                    recipient[validation],
                    donor[validation],
                    clean_target[validation],
                    delta[validation],
                ),
            )
            np.testing.assert_array_equal(
                decoder.predict(predicted), reloaded_decoder.predict(predicted)
            )


if __name__ == "__main__":
    unittest.main()
