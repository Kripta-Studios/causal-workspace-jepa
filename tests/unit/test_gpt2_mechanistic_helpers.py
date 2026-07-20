from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.gpt2_medium_mechanistic_study import (
    _build_split_masks,
    _intervention_features,
    _jacobian_predictions,
    _validate_cpu_limits,
)
from causal_workspace_jepa.models.intervention_jepa import RidgeInterventionPredictor


class GPT2MechanisticHelperTests(unittest.TestCase):
    def test_linear_and_bilinear_predictors_use_distinct_designs(self) -> None:
        rng = np.random.default_rng(8)
        context = rng.normal(size=(160, 3)).astype(np.float32)
        intervention = rng.normal(size=(160, 2)).astype(np.float32)
        target = (context[:, :1] * intervention[:, :1]).astype(np.float32)
        train = np.arange(120)
        test = np.arange(120, 160)
        linear = RidgeInterventionPredictor.fit(
            context[train], intervention[train], target[train], mode="linear"
        )
        bilinear = RidgeInterventionPredictor.fit(
            context[train], intervention[train], target[train], mode="bilinear"
        )
        linear_mse = np.mean(
            (linear.predict(context[test], intervention[test]) - target[test]) ** 2
        )
        bilinear_mse = np.mean(
            (bilinear.predict(context[test], intervention[test]) - target[test]) ** 2
        )
        self.assertLess(bilinear_mse, linear_mse * 0.01)

    def test_local_jacobian_uses_same_prompt_and_site(self) -> None:
        prompts = np.repeat(np.arange(2), 4)
        layers = np.full(8, 6)
        directions = np.zeros(8, dtype=int)
        magnitudes = np.tile(np.array([-1.0, -0.25, 0.25, 1.0]), 2)
        slopes = np.where(prompts == 0, 2.0, 3.0)
        targets = (slopes * magnitudes)[:, None].astype(np.float32)
        mask = np.isclose(magnitudes, 1.0)
        predicted = _jacobian_predictions(
            targets,
            prompts,
            layers,
            directions,
            magnitudes,
            mask,
            epsilon=0.25,
            local=True,
            train_prompt_count=1,
            train_layers=[6],
        )
        np.testing.assert_allclose(predicted, targets[mask])

    def test_intervention_encoding_and_resource_limits(self) -> None:
        encoded = _intervention_features(12, 1, -0.5, [6, 12, 18], 2)
        np.testing.assert_allclose(encoded, [0.0, 1.0, 0.0, 0.0, 1.0, -0.5])
        _validate_cpu_limits(64, [0, 1, 2], ["train", "test"], 1)
        with self.assertRaises(RuntimeError):
            _validate_cpu_limits(65, [0], ["train", "test"], 1)

    def test_prompt_magnitude_and_layer_splits_do_not_overlap(self) -> None:
        rows = [
            (prompt, layer, direction, magnitude)
            for layer in [6, 12, 18]
            for direction in range(2)
            for magnitude in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
            for prompt in range(8)
        ]
        prompts = np.asarray([row[0] for row in rows])
        layers = np.asarray([row[1] for row in rows])
        magnitudes = np.asarray([row[3] for row in rows])
        train, prompt_test, layer_test = _build_split_masks(
            prompts,
            layers,
            magnitudes,
            train_prompt_count=6,
            train_layers=[6, 12],
            heldout_layer=18,
            max_train_magnitude=0.5,
            test_magnitude=1.0,
        )
        self.assertEqual(int(train.sum()), 96)
        self.assertEqual(int(prompt_test.sum()), 16)
        self.assertEqual(int(layer_test.sum()), 8)
        self.assertFalse(np.any(train & prompt_test))
        self.assertFalse(np.any(train & layer_test))


if __name__ == "__main__":
    unittest.main()
