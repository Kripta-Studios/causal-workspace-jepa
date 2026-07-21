from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.hooks.names import transformer_site


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "optional torch/transformers dependencies are unavailable",
)
class QwenHFAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from causal_workspace_jepa.adapters.qwen_hf_adapter import (
            QwenAdapterConfig,
            QwenHFAdapter,
        )

        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
        )
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(config)
        cls.adapter = QwenHFAdapter(
            QwenAdapterConfig(
                model_name="tiny-random-qwen3",
                device="cpu",
                dtype="float32",
                max_length=8,
            ),
            model=model,
            tokenizer=_TinyTokenizer(),
        )
        cls.autograd_adapter = QwenHFAdapter(
            QwenAdapterConfig(
                model_name="tiny-random-qwen3",
                device="cpu",
                dtype="float32",
                max_length=8,
                preserve_autograd=True,
            ),
            model=model,
            tokenizer=_TinyTokenizer(),
        )

    def test_site_names_and_selected_capture(self) -> None:
        sites = [transformer_site(0, "resid_pre"), transformer_site(0, "attn_out"), "logits"]
        run = self.adapter.forward_with_cache(self.adapter.tokenize(["alpha beta"]), sites)
        self.assertEqual(set(run.activations), set(sites))
        self.assertEqual(run.activations[sites[0]].shape, (1, 2, 32))

    def test_hooked_clean_forward_is_deterministic(self) -> None:
        batch = self.adapter.tokenize(["alpha beta"])
        first = self.adapter.forward_with_cache(batch, [transformer_site(1, "resid_post")])
        second = self.adapter.forward_with_cache(batch, [transformer_site(0, "mlp_out")])
        np.testing.assert_allclose(first.logits, second.logits, atol=0.0, rtol=0.0)

    def test_direct_intervention_changes_logits(self) -> None:
        batch = self.adapter.tokenize(["alpha beta"])
        clean = self.adapter.forward_with_cache(batch, ["logits"])
        spec = InterventionSpec(
            site=transformer_site(0, "mlp_out"),
            operation="steer",
            positions=(1,),
            feature_ids=(0,),
            magnitude=2.0,
        )
        changed = self.adapter.forward_with_intervention(batch, spec, ["logits"])
        self.assertGreater(float(np.max(np.abs(changed.logits - clean.logits))), 0.0)

    def test_batch_one_mean_and_resample_require_registered_values(self) -> None:
        batch = self.adapter.tokenize(["alpha beta"])
        site = transformer_site(0, "resid_pre")
        with self.assertRaises(ValueError):
            self.adapter.forward_with_intervention(
                batch,
                InterventionSpec(site=site, operation="mean"),
                ["logits"],
            )
        with self.assertRaises(ValueError):
            self.adapter.forward_with_intervention(
                batch,
                InterventionSpec(site=site, operation="resample"),
                ["logits"],
            )

    def test_autograd_preserves_selected_activation(self) -> None:
        import torch

        site = transformer_site(0, "resid_post")
        run = self.autograd_adapter.forward_with_cache(
            self.autograd_adapter.tokenize(["alpha beta"]),
            [site],
        )
        gradient = torch.autograd.grad(run.logits.sum(), run.activations[site])[0]
        self.assertEqual(tuple(gradient.shape), (1, 2, 32))

    def test_directional_jvp_matches_symmetric_difference(self) -> None:
        import torch

        from causal_workspace_jepa.experiments.llm.qwen_jvp_audit import (
            _directional_output_function,
        )

        self.autograd_adapter.model.set_attn_implementation("eager")
        batch = self.autograd_adapter.tokenize(["alpha beta"])
        generator = torch.Generator().manual_seed(17)
        direction = torch.randn(32, generator=generator)
        projection = torch.randn(32, 4, generator=generator)
        logit_ids = torch.tensor([2, 3, 5], dtype=torch.long)
        function = _directional_output_function(
            self.autograd_adapter,
            batch,
            source_site=transformer_site(0, "resid_post"),
            target_site=transformer_site(1, "resid_post"),
            direction=direction,
            hidden_projection=projection,
            logit_ids=logit_ids,
        )
        scale = torch.zeros((), requires_grad=True)
        _, exact = torch.autograd.functional.jvp(
            function,
            scale,
            torch.ones_like(scale),
            strict=True,
        )
        epsilon = 1e-3
        central = (
            function(torch.tensor(epsilon)) - function(torch.tensor(-epsilon))
        ) / (2 * epsilon)
        torch.testing.assert_close(exact, central, atol=2e-3, rtol=2e-3)

    def test_full_selected_logit_jacobian_reconstructs_directional_jvp(self) -> None:
        import torch

        from causal_workspace_jepa.experiments.llm.qwen_context_geometry_study import (
            _selected_logit_jacobian,
        )
        from causal_workspace_jepa.experiments.llm.qwen_jvp_audit import (
            _directional_output_function,
        )

        self.autograd_adapter.model.set_attn_implementation("eager")
        batch = self.autograd_adapter.tokenize(["alpha beta"])
        source_site = transformer_site(0, "resid_post")
        target_site = transformer_site(1, "resid_post")
        run = self.autograd_adapter.forward_with_cache(batch, [source_site, "logits"])
        source = run.activations[source_site][0, -1].detach().float()
        logit_ids = torch.tensor([2, 3, 5], dtype=torch.long)
        jacobian, replay = _selected_logit_jacobian(
            self.autograd_adapter,
            batch,
            source_site=source_site,
            source=source,
            logit_ids=logit_ids,
        )
        torch.testing.assert_close(replay, run.logits[0, -1, logit_ids], atol=0.0, rtol=0.0)
        generator = torch.Generator().manual_seed(31)
        direction = torch.randn(32, generator=generator)
        function = _directional_output_function(
            self.autograd_adapter,
            batch,
            source_site=source_site,
            target_site=target_site,
            direction=direction,
            hidden_projection=torch.zeros((32, 0)),
            logit_ids=logit_ids,
        )
        scale = torch.zeros((), requires_grad=True)
        _, directional = torch.autograd.functional.jvp(
            function, scale, torch.ones_like(scale), strict=True
        )
        torch.testing.assert_close(jacobian @ direction, directional, atol=1e-5, rtol=1e-5)


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompts: list[str], **_kwargs: object) -> dict[str, object]:
        import torch

        rows = [[2 + (sum(map(ord, token)) % 61) for token in prompt.split()] for prompt in prompts]
        width = max(map(len, rows))
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"tok_{value}" for value in ids]


if __name__ == "__main__":
    unittest.main()
