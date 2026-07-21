from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_intervention_dataset import (
    _edited_source,
    _intervention_rows,
    qwen_intervention_prompts,
)


class QwenInterventionDatasetTests(unittest.TestCase):
    def test_prompt_splits_and_families_are_disjoint_by_id(self) -> None:
        prompts, splits, families = qwen_intervention_prompts()
        self.assertEqual(len(prompts), 12)
        self.assertEqual([splits.count(value) for value in (0, 1, 2)], [8, 2, 2])
        self.assertEqual(set(families), {"geography", "causal_physics"})
        self.assertEqual(len(set(prompts)), len(prompts))

    def test_registered_grid_has_expected_operations(self) -> None:
        rows = _intervention_rows(
            site="blocks.7.resid_post",
            steer_features=(0, 4),
            steer_magnitudes=(-2.0, 2.0),
            replacement_features=(0, 4),
            donor_patch=1,
            donor_resample=2,
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual({name for name, _ in rows}, {"steer", "zero", "mean", "patch", "resample"})

    def test_source_edit_matches_spec_without_downstream_target(self) -> None:
        rows = _intervention_rows(
            site="blocks.7.resid_post",
            steer_features=(0,),
            steer_magnitudes=(2.0,),
            replacement_features=(0, 2),
            donor_patch=1,
            donor_resample=2,
        )
        clean = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        mean = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)
        donor = np.asarray([7.0, 8.0, 9.0], dtype=np.float32)
        edited = {
            name: _edited_source(clean, spec, mean=mean, donor=donor)
            for name, spec in rows
        }
        self.assertEqual(float(edited["steer"][0]), 3.0)
        np.testing.assert_array_equal(edited["zero"], [0.0, 2.0, 0.0])
        np.testing.assert_array_equal(edited["mean"], [4.0, 2.0, 6.0])
        np.testing.assert_array_equal(edited["patch"], [7.0, 2.0, 9.0])


if __name__ == "__main__":
    unittest.main()
