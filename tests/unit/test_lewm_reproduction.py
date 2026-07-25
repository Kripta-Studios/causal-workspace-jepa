from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - minimal CPU install
    torch = None

from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.data.synthetic.pixel_tiny_maze import generate_pixel_tiny_maze


@unittest.skipIf(torch is None, "torch is optional")
class SmallLeWorldModelTests(unittest.TestCase):
    def _model(self):
        from causal_workspace_jepa.models.lewm import SmallLeWMConfig, SmallLeWorldModel

        return SmallLeWorldModel(
            SmallLeWMConfig(
                image_size=8,
                patch_size=4,
                latent_dim=8,
                encoder_depth=1,
                predictor_depth=1,
                heads=2,
                mlp_dim=16,
                max_history=2,
                sigreg_projections=4,
            )
        )

    def test_two_loss_forward_backward_and_checkpoint_replay(self) -> None:
        torch.manual_seed(3)
        model = self._model().eval()
        pixels = torch.rand(5, 2, 3, 8, 8)
        actions = torch.nn.functional.one_hot(torch.arange(5) % 4, 4).float()[:, None]
        losses = model.loss(pixels, actions)
        self.assertEqual(set(losses), {"loss", "prediction_loss", "sigreg_loss"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["loss"].backward()
        self.assertIsNotNone(model.encoder.patch.weight.grad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            restored = type(model).load(path).eval()
            with torch.inference_mode():
                expected = model.forward_sequence(pixels, actions)["predicted_embeddings"]
                observed = restored.forward_sequence(pixels, actions)["predicted_embeddings"]
            torch.testing.assert_close(expected, observed, rtol=0, atol=0)

    def test_adapter_protocol_and_replayable_intervention(self) -> None:
        from causal_workspace_jepa.adapters.lewm_adapter import LeWorldModelAdapter

        torch.manual_seed(5)
        model = self._model().eval()
        adapter = LeWorldModelAdapter(model)
        pixels = np.random.default_rng(5).random((6, 3, 8, 8), dtype=np.float32)
        embeddings = adapter.encode(pixels).tensor
        states = np.arange(12, dtype=np.float32).reshape(6, 2)
        adapter.fit_state_decoder(embeddings, states)
        actions = np.zeros((6, 1, 4), dtype=np.float32)
        actions[:, :, 1] = 1.0
        clean = adapter.predict(adapter.encode(pixels), actions, return_intermediates=True)
        zero = adapter.predict_with_intervention(
            adapter.encode(pixels),
            actions,
            InterventionSpec(site="predictor.block0", operation="zero"),
            return_intermediates=True,
        )
        self.assertEqual(clean.predicted_latents.shape, (6, 1, 8))
        self.assertIn("predictor.block0", clean.intermediates)
        self.assertIsNotNone(clean.decoded_state)
        self.assertGreater(
            float(np.max(np.abs(clean.predicted_latents - zero.predicted_latents))), 0.0
        )

    def test_adapter_intervention_positions_select_rollout_steps(self) -> None:
        from causal_workspace_jepa.adapters.lewm_adapter import LeWorldModelAdapter

        torch.manual_seed(13)
        adapter = LeWorldModelAdapter(self._model().eval())
        pixels = np.random.default_rng(13).random((2, 3, 8, 8), dtype=np.float32)
        latent = adapter.encode(pixels)
        actions = np.zeros((2, 3, 4), dtype=np.float32)
        actions[:, :, 1] = 1.0
        clean = adapter.predict(latent, actions, return_intermediates=True)
        at_zero = adapter.predict_with_intervention(
            latent,
            actions,
            InterventionSpec(site="predictor.block0", operation="zero", positions=(0,)),
            return_intermediates=True,
        )
        at_two = adapter.predict_with_intervention(
            latent,
            actions,
            InterventionSpec(site="predictor.block0", operation="zero", positions=(2,)),
            return_intermediates=True,
        )

        np.testing.assert_array_equal(
            at_zero.intermediates["predictor.block0"][:, 0],
            np.zeros((2, 8), dtype=np.float32),
        )
        np.testing.assert_allclose(
            at_two.intermediates["predictor.block0"][:, 0],
            clean.intermediates["predictor.block0"][:, 0],
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            at_two.intermediates["predictor.block0"][:, 2],
            np.zeros((2, 8), dtype=np.float32),
        )
        self.assertGreater(
            float(np.max(np.abs(at_zero.predicted_latents - at_two.predicted_latents))),
            0.0,
        )
        at_last = adapter.predict_with_intervention(
            latent,
            actions,
            InterventionSpec(site="predictor.block0", operation="zero", positions=(-1,)),
            return_intermediates=True,
        )
        np.testing.assert_array_equal(
            at_last.intermediates["predictor.block0"][:, 2],
            np.zeros((2, 8), dtype=np.float32),
        )
        with self.assertRaisesRegex(IndexError, "out of range"):
            adapter.predict_with_intervention(
                latent,
                actions,
                InterventionSpec(site="predictor.block0", operation="zero", positions=(99,)),
            )

    def test_adapter_project_out_uses_nonorthogonal_column_basis(self) -> None:
        from causal_workspace_jepa.adapters.lewm_adapter import LeWorldModelAdapter

        torch.manual_seed(19)
        adapter = LeWorldModelAdapter(self._model().eval())
        pixels = np.random.default_rng(19).random((2, 3, 8, 8), dtype=np.float32)
        latent = adapter.encode(pixels)
        actions = np.zeros((2, 1, 4), dtype=np.float32)
        actions[:, :, 2] = 1.0
        basis = np.zeros((8, 2), dtype=np.float32)
        basis[0, 0] = 1.0
        basis[0, 1] = 1.0
        basis[1, 1] = 1.0
        adapter.register_basis("predictor.block0", basis)

        projected = adapter.predict_with_intervention(
            latent,
            actions,
            InterventionSpec(site="predictor.block0", operation="project_out"),
            return_intermediates=True,
        )

        projected_block = projected.intermediates["predictor.block0"]
        np.testing.assert_allclose(projected_block @ basis, 0.0, atol=1e-5)


class PixelTinyMazeTests(unittest.TestCase):
    def test_generator_is_deterministic_and_action_aligned(self) -> None:
        first = generate_pixel_tiny_maze(trajectories=6, steps=4, seed=17)
        second = generate_pixel_tiny_maze(trajectories=6, steps=4, seed=17)
        np.testing.assert_array_equal(first.observations, second.observations)
        np.testing.assert_array_equal(first.actions, second.actions)
        self.assertEqual(first.observations.shape, (6, 4, 3, 20, 20))
        np.testing.assert_allclose(first.actions.sum(axis=-1), 1.0)


if __name__ == "__main__":
    unittest.main()
