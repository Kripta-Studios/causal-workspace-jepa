from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_binding_format_diagnostic import (
    compute_binding_format_diagnostic,
)
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_protocol import (
    BindingEpisode,
)


class QwenBindingFormatDiagnosticTests(unittest.TestCase):
    def test_exact_pairs_separate_full_and_four_value_accuracy(self) -> None:
        primary = BindingEpisode(
            episode_id="test-0000",
            split="test",
            keys=("a", "b", "c", "d"),
            recipient_values=("v0", "v1", "v2", "v3"),
            donor_values=("v1", "v0", "v2", "v3"),
            query_index=0,
            swapped_indices=(0, 1),
            template="primary",
        )
        paraphrase = BindingEpisode(
            episode_id="paraphrase-0000",
            split="paraphrase",
            keys=primary.keys,
            recipient_values=primary.recipient_values,
            donor_values=primary.donor_values,
            query_index=primary.query_index,
            swapped_indices=primary.swapped_indices,
            template="paraphrase",
        )
        value_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
        arrays = {
            "recipient_answer_id": np.asarray([10, 10]),
            "donor_answer_id": np.asarray([11, 11]),
            "clean_top_token": np.asarray([99, 10]),
            "donor_top_token": np.asarray([99, 11]),
            "treated_top_token": np.asarray([99, 11]),
            "clean_value_logits": np.asarray([[4, 3, 2, 1], [6, 1, 0, -1]], dtype=float),
            "donor_value_logits": np.asarray([[3, 4, 2, 1], [1, 6, 0, -1]], dtype=float),
            "treated_value_logits": np.asarray([[3, 4, 2, 1], [1, 6, 0, -1]], dtype=float),
        }
        result = compute_binding_format_diagnostic(
            arrays,
            [primary, paraphrase],
            value_ids,
            value_token_by_string={"v0": 10, "v1": 11, "v2": 12, "v3": 13},
            decode_token=lambda token: {99: " ?\n"}.get(token, str(token)),
        )
        test = result["by_split"]["test"]["clean"]
        self.assertEqual(test["full_vocabulary_accuracy"], 0.0)
        self.assertEqual(test["episode_four_value_accuracy"], 1.0)
        self.assertEqual(test["dominant_top_token_decoded"], " ?\n")
        paired = result["paired_test_to_paraphrase"]["clean"]
        self.assertEqual(paired["full_gained_under_paraphrase"], 1)
        self.assertEqual(paired["full_lost_under_paraphrase"], 0)
        self.assertTrue(result["donor_treated_top_tokens_identical"])
        self.assertEqual(result["donor_treated_value_logits_max_abs_error"], 0.0)

    def test_nonfinite_and_unpaired_inputs_fail_closed(self) -> None:
        episode = BindingEpisode(
            episode_id="test-0000",
            split="test",
            keys=("a", "b", "c", "d"),
            recipient_values=("v0", "v1", "v2", "v3"),
            donor_values=("v1", "v0", "v2", "v3"),
            query_index=0,
            swapped_indices=(0, 1),
        )
        arrays = {
            "recipient_answer_id": np.asarray([10]),
            "donor_answer_id": np.asarray([11]),
            "clean_top_token": np.asarray([99]),
            "donor_top_token": np.asarray([99]),
            "treated_top_token": np.asarray([99]),
            "clean_value_logits": np.asarray([[np.nan, 3, 2, 1]]),
            "donor_value_logits": np.asarray([[3, 4, 2, 1]], dtype=float),
            "treated_value_logits": np.asarray([[3, 4, 2, 1]], dtype=float),
        }
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            compute_binding_format_diagnostic(
                arrays,
                [episode],
                [10, 11, 12, 13],
                value_token_by_string={"v0": 10, "v1": 11, "v2": 12, "v3": 13},
                decode_token=str,
            )
        arrays["clean_value_logits"][0, 0] = 4.0
        with self.assertRaisesRegex(ValueError, "equally sized"):
            compute_binding_format_diagnostic(
                arrays,
                [episode],
                [10, 11, 12, 13],
                value_token_by_string={"v0": 10, "v1": 11, "v2": 12, "v3": 13},
                decode_token=str,
            )

    def test_paired_answer_ids_and_logit_width_fail_closed(self) -> None:
        episodes = [
            BindingEpisode(
                episode_id=f"{split}-0000",
                split=split,
                keys=("a", "b", "c", "d"),
                recipient_values=("v0", "v1", "v2", "v3"),
                donor_values=("v1", "v0", "v2", "v3"),
                query_index=0,
                swapped_indices=(0, 1),
                template="primary" if split == "test" else "paraphrase",
            )
            for split in ("test", "paraphrase")
        ]
        arrays = {
            "recipient_answer_id": np.asarray([10, 12]),
            "donor_answer_id": np.asarray([11, 11]),
            "clean_top_token": np.asarray([10, 10]),
            "donor_top_token": np.asarray([11, 11]),
            "treated_top_token": np.asarray([11, 11]),
            "clean_value_logits": np.ones((2, 4)),
            "donor_value_logits": np.ones((2, 4)),
            "treated_value_logits": np.ones((2, 4)),
        }
        kwargs = {
            "value_token_by_string": {"v0": 10, "v1": 11, "v2": 12, "v3": 13},
            "decode_token": str,
        }
        with self.assertRaisesRegex(ValueError, "answer IDs"):
            compute_binding_format_diagnostic(arrays, episodes, [10, 11, 12, 13], **kwargs)
        arrays["recipient_answer_id"][1] = 10
        arrays["clean_value_logits"] = np.ones((2, 3))
        with self.assertRaisesRegex(ValueError, "widths"):
            compute_binding_format_diagnostic(arrays, episodes, [10, 11, 12, 13], **kwargs)


if __name__ == "__main__":
    unittest.main()
