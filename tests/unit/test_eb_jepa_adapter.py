from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - minimal CPU install
    torch = None

from causal_workspace_jepa.common.types import InterventionSpec, LatentState


@unittest.skipIf(torch is None, "torch is optional")
class EBJEPAAdapterTests(unittest.TestCase):
    @staticmethod
    def _model():
        class Encoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(2, 5)

            def forward(self, observations):
                # Preserve the official [B,D,T,1,1] representation contract.
                pooled = observations.mean(dim=(-1, -2)).transpose(1, 2)
                return self.projection(pooled).transpose(1, 2)[..., None, None]

        class Predictor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = torch.nn.GRU(input_size=2, hidden_size=5, num_layers=1)
                self.final_ln = torch.nn.LayerNorm(5)

            def forward(self, state, action):
                hidden = state.flatten(1, 4).unsqueeze(0).contiguous()
                inputs = action.squeeze(-1).unsqueeze(0).contiguous()
                next_state, _ = self.rnn(inputs, hidden)
                next_state = self.final_ln(next_state)
                return next_state[0, ..., None, None, None]

        class ContractModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = Encoder()
                self.action_encoder = torch.nn.Identity()
                self.predictor = Predictor()

        torch.manual_seed(17)
        return ContractModel()

    def test_exact_gru_decomposition_and_shapes(self) -> None:
        from causal_workspace_jepa.adapters.eb_jepa_adapter import EBJEPAAdapter

        adapter = EBJEPAAdapter(self._model())
        pixels = np.random.default_rng(7).normal(size=(3, 2, 2, 6, 6)).astype(np.float32)
        latent = adapter.encode(pixels)
        self.assertEqual(latent.tensor.shape, (3, 2, 5))
        actions = np.random.default_rng(11).normal(size=(3, 4, 2)).astype(np.float32)
        error = adapter.validate_predictor_reconstruction(latent, actions, atol=1e-6)
        self.assertLessEqual(error, 1e-6)
        output = adapter.predict(latent, actions, return_intermediates=True)
        self.assertEqual(output.predicted_latents.shape, (3, 4, 5))
        self.assertEqual(output.action_embeddings.shape, (3, 4, 2))
        self.assertEqual(output.intermediates["predictor.update_gate"].shape, (3, 4, 5))
        np.testing.assert_allclose(
            output.predicted_latents,
            output.intermediates["predictor.hidden_post_norm"],
            rtol=0,
            atol=0,
        )

    def test_gate_intervention_is_position_and_feature_specific(self) -> None:
        from causal_workspace_jepa.adapters.eb_jepa_adapter import EBJEPAAdapter

        adapter = EBJEPAAdapter(self._model())
        latent = LatentState(np.arange(10, dtype=np.float32).reshape(2, 5) / 10)
        actions = np.ones((2, 3, 2), dtype=np.float32)
        clean = adapter.predict(latent, actions, return_intermediates=True)
        edited = adapter.predict_with_intervention(
            latent,
            actions,
            InterventionSpec(
                site="predictor.update_gate",
                operation="zero",
                positions=(1,),
                feature_ids=(0,),
            ),
            return_intermediates=True,
        )
        np.testing.assert_allclose(
            edited.intermediates["predictor.update_gate"][:, 0],
            clean.intermediates["predictor.update_gate"][:, 0],
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            edited.intermediates["predictor.update_gate"][:, 1, 0],
            np.zeros(2, dtype=np.float32),
        )
        np.testing.assert_allclose(
            edited.intermediates["predictor.update_gate"][:, 1, 1:],
            clean.intermediates["predictor.update_gate"][:, 1, 1:],
            rtol=0,
            atol=0,
        )
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        edited.intermediates["predictor.update_gate"][:, 2]
                        - clean.intermediates["predictor.update_gate"][:, 2]
                    )
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(np.max(np.abs(edited.predicted_latents - clean.predicted_latents))), 0.0
        )

    def test_linear_decoder_and_invalid_multilayer_contract(self) -> None:
        from causal_workspace_jepa.adapters.eb_jepa_adapter import EBJEPAAdapter

        adapter = EBJEPAAdapter(self._model())
        rng = np.random.default_rng(13)
        embeddings = rng.normal(size=(12, 5)).astype(np.float32)
        states = embeddings[:, :2] * 2.0
        adapter.fit_state_decoder(embeddings, states)
        decoded = adapter.decode_state(LatentState(embeddings))["state"]
        self.assertEqual(decoded.shape, states.shape)
        self.assertLess(float(np.mean(np.square(decoded - states))), 1e-5)

        invalid = self._model()
        invalid.predictor.rnn = torch.nn.GRU(2, 5, num_layers=2)
        with self.assertRaisesRegex(ValueError, "one-layer"):
            EBJEPAAdapter(invalid)


if __name__ == "__main__":
    unittest.main()
