from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_capital_patch_dataset import (
    CAPITAL_PAIRS,
    SPLIT_COUNTRIES,
    capital_split_ids,
)


class QwenCapitalPatchDatasetTests(unittest.TestCase):
    def test_entity_splits_are_disjoint_and_complete(self) -> None:
        countries = {country for country, _capital in CAPITAL_PAIRS}
        split_sets = [set(values) for values in SPLIT_COUNTRIES.values()]
        self.assertEqual([len(values) for values in split_sets], [24, 6, 6])
        self.assertFalse(split_sets[0] & split_sets[1])
        self.assertFalse(split_sets[0] & split_sets[2])
        self.assertFalse(split_sets[1] & split_sets[2])
        self.assertEqual(set.union(*split_sets), countries)

    def test_pair_grid_has_registered_size(self) -> None:
        expected = sum(len(values) * (len(values) - 1) for values in SPLIT_COUNTRIES.values())
        self.assertEqual(expected, 612)
        split_ids = capital_split_ids()
        np.testing.assert_array_equal(np.bincount(split_ids), [24, 6, 6])

    def test_calibration_entities_are_excluded(self) -> None:
        countries = {country for country, _capital in CAPITAL_PAIRS}
        self.assertTrue({"Japan", "Canada", "China", "Kenya"}.isdisjoint(countries))


if __name__ == "__main__":
    unittest.main()
