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
