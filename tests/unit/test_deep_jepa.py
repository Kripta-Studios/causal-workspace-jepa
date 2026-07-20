from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.interpretability.workspace_tests import (
    activation_manifold_diagnostics,
    conditional_resample_subspace,
    counterfactual_subspace_audit,
    local_tangent_subspaces,
)
from causal_workspace_jepa.models.deep_jepa import DeepActionConditionedJEPA
from causal_workspace_jepa.models.uncertainty import IntervalCalibrator, rank_auc


class DeepJepaTests(unittest.TestCase):
    def test_learned_predictor_exposes_real_hidden_sites(self) -> None:
        dataset = generate_pointmass2d(trajectories=24, steps=12, seed=8)
        model = DeepActionConditionedJEPA.fit(
            dataset.observations,
            dataset.actions,
            latent_dim=6,
            hidden_dims=(12, 8),
            training_steps=400,
            batch_size=64,
            seed=8,
            encoder_seed=3,
        )
        latent = model.encode(dataset.observations[:3, 0])
        output = model.predict(latent, dataset.actions[:3, :2], return_intermediates=True)
        self.assertEqual(output.predicted_latents.shape, (3, 2, 6))
        self.assertEqual(output.intermediates["predictor.hidden1"].shape, (3, 2, 12))
        self.assertEqual(output.intermediates["predictor.hidden2"].shape, (3, 2, 8))
        self.assertIn("predictor.hidden2", model.named_activation_points())
        target = model.encode(dataset.observations[:3, 1:3]).tensor
        self.assertLess(float(np.mean((output.predicted_latents - target) ** 2)), 0.05)

    def test_conditional_resampling_uses_empirical_donors(self) -> None:
        bank = np.array(
            [[0.0, 0.0], [1.0, 0.1], [2.0, 0.2], [3.0, 0.3]],
            dtype=np.float32,
        )
        recipients = np.array([[1.4, 0.15], [2.4, 0.25]], dtype=np.float32)
        basis = np.array([[1.0], [0.0]], dtype=np.float32)
        patched, donor_ids = conditional_resample_subspace(
            recipients,
            bank,
            basis,
            neighbor_pool=2,
        )
        np.testing.assert_allclose(patched[:, 1], recipients[:, 1])
        self.assertTrue(np.all((0 <= donor_ids) & (donor_ids < len(bank))))
        diagnostics = activation_manifold_diagnostics(recipients, patched, bank)
        self.assertGreater(diagnostics["perturbation_rms"], 0.0)
        self.assertTrue(np.isfinite(diagnostics["median_density_distance_ratio"]))

    def test_split_calibrator_and_auc(self) -> None:
        targets = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        predictions = np.stack([targets - 0.2, targets, targets + 0.2], axis=0)
        calibrator = IntervalCalibrator.fit(predictions, targets, target_coverage=0.9)
        report = calibrator.evaluate(predictions, targets)
        self.assertGreaterEqual(report["coverage"], 0.75)
        self.assertAlmostEqual(rank_auc(np.array([0.0, 0.1]), np.array([0.9, 1.0])), 1.0)

    def test_tangent_controls_and_counterfactual_swap(self) -> None:
        rng = np.random.default_rng(12)
        bank = rng.normal(size=(80, 6)).astype(np.float32)
        controls = local_tangent_subspaces(
            bank,
            2,
            count=8,
            neighbors=10,
            seed=12,
        )
        self.assertEqual(len(controls), 8)
        for basis in controls:
            np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-5)
        recipients = bank[:16]
        donors = recipients.copy()
        donors[:, :2] += 0.25
        candidate = np.eye(6, dtype=np.float32)[:, :2]
        consumers = {"readout": lambda value: value[:, :2]}
        audit = counterfactual_subspace_audit(
            recipients,
            donors,
            bank,
            consumers,
            candidate,
            controls,
            min_matched_controls=1,
            match_factor=10.0,
        )
        self.assertAlmostEqual(audit["candidate"]["mean_recovery"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
