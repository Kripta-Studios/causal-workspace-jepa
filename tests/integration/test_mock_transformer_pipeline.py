from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.adapters.mock_transformer import (
    MockInstrumentedCausalLM,
    MockTransformerConfig,
    build_mock_prompts,
)
from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.hooks.names import transformer_site
from causal_workspace_jepa.models.intervention_jepa import (
    LayerTransitionInterventionJEPA,
    effect_correlation,
    no_change_mse,
)


class MockTransformerPipelineTests(unittest.TestCase):
    def test_direct_intervention_changes_downstream_logits(self) -> None:
        model = MockInstrumentedCausalLM(MockTransformerConfig(seed=0))
        batch = model.tokenize(build_mock_prompts(4))
        sites = [transformer_site(1, "mlp_out"), transformer_site(2, "resid_post"), "logits"]
        clean = model.forward_with_cache(batch, sites)
        spec = InterventionSpec(
            site=transformer_site(1, "mlp_out"),
            operation="steer",
            positions=(2,),
            feature_ids=(0,),
            magnitude=1.0,
            seed=0,
        )
        intervened = model.forward_with_intervention(batch, spec, sites)
        self.assertGreater(float(np.linalg.norm(intervened.logits - clean.logits)), 0.0)

    def test_intervention_jepa_fits_known_effects_without_post_target_context(self) -> None:
        contexts = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        interventions = np.array([[1.0], [1.0], [-1.0], [-1.0]], dtype=np.float32)
        targets = 2.0 * interventions + 0.5 * contexts * interventions
        model = LayerTransitionInterventionJEPA.fit(contexts, interventions, targets)
        prediction = model.predict(contexts, interventions)
        self.assertLess(np.mean((prediction - targets) ** 2), no_change_mse(targets))
        self.assertGreater(effect_correlation(prediction, targets), 0.99)


if __name__ == "__main__":
    unittest.main()
