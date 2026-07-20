from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.gpt2_medium_mechanistic_study import _build_split_masks


class GPT2SplitScientificControls(unittest.TestCase):
    def test_meta_model_training_excludes_evaluation_prompts_magnitudes_and_layer(self) -> None:
        rows = [
            (prompt, layer, magnitude)
            for layer in [6, 12, 18]
            for _direction in range(2)
            for magnitude in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
            for prompt in range(8)
        ]
        prompts = np.asarray([row[0] for row in rows])
        layers = np.asarray([row[1] for row in rows])
        magnitudes = np.asarray([row[2] for row in rows])
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
        self.assertFalse(np.any(train & (prompt_test | layer_test)))
        self.assertTrue(np.all(prompts[train] < 6))
        self.assertTrue(np.all(np.abs(magnitudes[train]) <= 0.5))
        self.assertTrue(np.all(layers[layer_test] == 18))


if __name__ == "__main__":
    unittest.main()
