from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.gpt2_medium_semantic_composition_study import (
    _additive_jacobian_predictions,
    _build_semantic_split_masks,
    _composition_nonlinearity,
    _construct_semantic_directions,
    _intervention_grid,
    _large_additive_predictions,
)


class GPT2SemanticHelperTests(unittest.TestCase):
    def test_contrast_directions_are_normalized_and_orthogonal(self) -> None:
        import torch

        activations = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0, 0.0],
            ]
        )
        directions, report = _construct_semantic_directions(activations)
        torch.testing.assert_close(directions @ directions.T, torch.eye(2))
        self.assertAlmostEqual(report["orthogonalized_cosine"], 0.0)

    def test_grid_and_splits_hold_out_compositions(self) -> None:
        grid = _intervention_grid(0.5, 6.0)
        self.assertEqual(len(grid), 12)
        self.assertEqual(sum(operation == "single" for operation, _ in grid), 8)
        self.assertEqual(sum(operation == "composition" for operation, _ in grid), 4)
        prompts = np.tile(np.arange(6), len(grid))
        operations = np.repeat(
            [int(operation == "composition") for operation, _ in grid],
            6,
        )
        train, primary, seen = _build_semantic_split_masks(
            prompts,
            operations,
            train_prompt_count=4,
        )
        self.assertEqual(int(train.sum()), 32)
        self.assertEqual(int(primary.sum()), 8)
        self.assertEqual(int(seen.sum()), 16)
        self.assertFalse(np.any(train & primary))

    def test_additive_baselines_recover_linear_effects_and_expose_interaction(self) -> None:
        grid = _intervention_grid(0.5, 6.0)
        prompts = np.tile(np.arange(2), len(grid))
        operations = np.repeat(
            [int(operation == "composition") for operation, _ in grid],
            2,
        )
        coefficients = np.repeat(
            np.stack([coefficient for _, coefficient in grid]),
            2,
            axis=0,
        )
        linear = 2.0 * coefficients[:, :1] - 3.0 * coefficients[:, 1:2]
        interaction = (
            operations[:, None] * 0.25 * coefficients[:, :1] * coefficients[:, 1:2]
        )
        targets = (linear + interaction).astype(np.float32)
        composition_mask = operations == 1
        jacobian = _additive_jacobian_predictions(
            targets,
            prompts,
            operations,
            coefficients,
            composition_mask,
            epsilon=0.5,
            local=True,
            train_prompt_count=1,
        )
        direct_additive = _large_additive_predictions(
            targets,
            prompts,
            operations,
            coefficients,
            composition_mask,
        )
        np.testing.assert_allclose(jacobian, linear[composition_mask])
        np.testing.assert_allclose(direct_additive, linear[composition_mask])
        report = _composition_nonlinearity(direct_additive, targets[composition_mask])
        self.assertGreater(report["interaction_fraction_of_effect"], 0.0)


if __name__ == "__main__":
    unittest.main()
