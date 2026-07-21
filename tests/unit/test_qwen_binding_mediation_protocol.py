from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.common.config import load_config

from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_protocol import (
    BindingEpisode,
    MediationEstimate,
    assert_disjoint_pools,
    audit_tokenized_treatment,
    clustered_bootstrap_mediation,
    generate_binding_episodes,
    mediation_estimate,
    monte_carlo_upper_tail_p,
    select_train_prefix,
)


class QwenBindingMediationProtocolTests(unittest.TestCase):
    def test_episode_generator_is_deterministic_balanced_and_duplicate_free(self) -> None:
        kwargs = {
            "split": "train",
            "keys": tuple(f"k{i}" for i in range(8)),
            "values": tuple(f"v{i}" for i in range(8)),
            "count": 24,
            "seed": 401,
        }
        first = generate_binding_episodes(**kwargs)
        second = generate_binding_episodes(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len({episode.episode_id for episode in first}), 24)
        self.assertEqual(
            [sum(episode.query_index == index for episode in first) for index in range(4)],
            [6, 6, 6, 6],
        )
        for episode in first:
            self.assertNotEqual(episode.recipient_answer, episode.donor_answer)

    def test_episode_rejects_non_transposition_and_unchanged_query(self) -> None:
        with self.assertRaises(ValueError):
            BindingEpisode(
                episode_id="bad",
                split="train",
                keys=("a", "b", "c", "d"),
                recipient_values=("w", "x", "y", "z"),
                donor_values=("x", "w", "y", "z"),
                query_index=2,
                swapped_indices=(0, 1),
            )

    def test_token_audit_accepts_exact_two_token_swap(self) -> None:
        episode = BindingEpisode(
            episode_id="train-0000",
            split="train",
            keys=("a", "b", "c", "d"),
            recipient_values=("w", "x", "y", "z"),
            donor_values=("x", "w", "y", "z"),
            query_index=0,
            swapped_indices=(0, 1),
        )
        prompt_ids = {
            episode.recipient_prompt(): [1, 10, 2, 11, 3],
            episode.donor_prompt(): [1, 11, 2, 10, 3],
        }
        answer_ids = {"w": [10], "x": [11]}
        audit = audit_tokenized_treatment(
            episode,
            encode_prompt=lambda prompt: prompt_ids[prompt],
            encode_answer=lambda answer: answer_ids[answer],
        )
        self.assertEqual(audit.changed_positions, (1, 3))
        self.assertEqual(audit.recipient_answer_id, 10)
        self.assertEqual(audit.donor_answer_id, 11)

    def test_token_audit_rejects_extra_context_change(self) -> None:
        episode = BindingEpisode(
            episode_id="train-0000",
            split="train",
            keys=("a", "b", "c", "d"),
            recipient_values=("w", "x", "y", "z"),
            donor_values=("x", "w", "y", "z"),
            query_index=0,
            swapped_indices=(0, 1),
        )
        with self.assertRaises(ValueError):
            audit_tokenized_treatment(
                episode,
                encode_prompt=lambda prompt: (
                    [1, 11, 2, 10, 4]
                    if prompt == episode.donor_prompt()
                    else [1, 10, 2, 11, 3]
                ),
                encode_answer=lambda answer: [10 if answer == "w" else 11],
            )

    def test_ratio_of_sums_mediation(self) -> None:
        estimate = mediation_estimate(
            clean_scores=[0.0, 0.0],
            treated_scores=[2.0, 4.0],
            sufficient_scores=[1.0, 2.0],
            restored_scores=[0.5, 1.0],
        )
        self.assertTrue(estimate.eligible)
        self.assertAlmostEqual(estimate.sufficiency, 0.5)
        self.assertAlmostEqual(estimate.necessity, 0.75)

    def test_zero_aggregate_effect_is_ineligible(self) -> None:
        estimate = mediation_estimate([0.0, 0.0], [1.0, -1.0], [0.0, 0.0], [0.0, 0.0])
        self.assertFalse(estimate.eligible)
        self.assertTrue(np.isnan(estimate.sufficiency))

    def test_clustered_bootstrap_is_seed_reproducible(self) -> None:
        kwargs = {
            "clean_scores": np.zeros(16),
            "treated_scores": np.arange(1.0, 17.0),
            "sufficient_scores": np.arange(1.0, 17.0) * 0.7,
            "restored_scores": np.arange(1.0, 17.0) * 0.4,
            "draws": 128,
            "seed": 409,
        }
        self.assertEqual(clustered_bootstrap_mediation(**kwargs), clustered_bootstrap_mediation(**kwargs))

    def test_prefix_selection_uses_smallest_jointly_eligible_prefix(self) -> None:
        estimates = {
            1: MediationEstimate(1.0, 0.7, 0.4, True),
            2: MediationEstimate(1.0, 0.65, 0.61, True),
            3: MediationEstimate(1.0, 0.8, 0.8, True),
        }
        selection = select_train_prefix(["a", "b", "c"], estimates)
        self.assertTrue(selection.eligible)
        self.assertEqual(selection.selected, ("a", "b"))

    def test_disjoint_pool_and_monte_carlo_controls(self) -> None:
        assert_disjoint_pools({"train": ["a", "b"], "test": ["c", "d"]})
        with self.assertRaises(ValueError):
            assert_disjoint_pools({"train": ["a"], "test": ["a"]})
        self.assertAlmostEqual(monte_carlo_upper_tail_p(0.5, [0.1, 0.6, 0.2]), 0.5)

    def test_registered_config_has_disjoint_pools_and_fixed_candidate_count(self) -> None:
        config = load_config("configs/experiments/qwen_binding_mediation_v1.yaml")
        assert_disjoint_pools(config["token_pools"]["keys"])
        assert_disjoint_pools(config["token_pools"]["values"])
        self.assertEqual(config["candidates"]["node_count"], 56)
        self.assertEqual(config["ranking"]["maximum_nodes"], 4)
        self.assertEqual(sum(split["count"] for split in config["splits"].values()), 560)


if __name__ == "__main__":
    unittest.main()
