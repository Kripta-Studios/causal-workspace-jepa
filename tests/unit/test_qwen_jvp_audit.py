from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_jvp_audit import (
    _regenerate_hidden_projection,
    _row_relative_error,
    deduplicated_effect_mask,
)


class QwenJVPAuditTests(unittest.TestCase):
    def test_projection_regeneration_is_deterministic(self) -> None:
        first = _regenerate_hidden_projection(
            hidden_size=16, context_dim=4, hidden_target_dim=5, seed=53
        )
        second = _regenerate_hidden_projection(
            hidden_size=16, context_dim=4, hidden_target_dim=5, seed=53
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (16, 5))

    def test_relative_error_uses_row_norms(self) -> None:
        target = np.asarray([[3.0, 4.0], [1.0, 0.0]])
        prediction = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(_row_relative_error(prediction, target), [1.0, 0.0])

    def test_deduplication_keeps_first_exact_effect(self) -> None:
        prompt = np.asarray([0, 0, 1])
        layer = np.asarray([14, 14, 14])
        delta = np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        effect = np.asarray([[3.0], [3.0], [3.0]])
        np.testing.assert_array_equal(
            deduplicated_effect_mask(prompt, layer, delta, effect),
            [True, False, True],
        )


if __name__ == "__main__":
    unittest.main()
