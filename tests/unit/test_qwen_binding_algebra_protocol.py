from __future__ import annotations

import copy
import unittest
from collections import Counter

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_protocol import (
    DOUBLE_TRANSPOSITION_CLASS,
    FOUR_CYCLE_CLASS,
    HELD_OUT_COMPOSITION_CLASSES,
    IDENTITY_CLASS,
    THREE_CYCLE_CLASS,
    TRANSPOSITION_CLASS,
    BindingAlgebraCase,
    all_s4_permutations,
    apply_permutation,
    assert_action_class_partition,
    assert_globally_disjoint_token_pools,
    binding_algebra_cases_from_config,
    binding_algebra_episodes_from_config,
    binding_algebra_protocol_digest,
    compose_permutations,
    compose_rollout,
    decompose_into_transpositions,
    generate_binding_algebra_episodes,
    identity_permutation,
    inverse_permutation,
    permutation_changes_slot,
    permutation_class,
    permutations_in_classes,
    transposition,
    transposition_generators,
    validate_permutation,
)


class QwenBindingAlgebraProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("configs/experiments/qwen_binding_algebra_v1.yaml")

    def test_s4_roster_and_cycle_class_counts_are_exact(self) -> None:
        actions = all_s4_permutations()
        self.assertEqual(len(actions), 24)
        self.assertEqual(len(set(actions)), 24)
        self.assertEqual(
            Counter(permutation_class(action) for action in actions),
            {
                IDENTITY_CLASS: 1,
                TRANSPOSITION_CLASS: 6,
                DOUBLE_TRANSPOSITION_CLASS: 3,
                THREE_CYCLE_CLASS: 8,
                FOUR_CYCLE_CLASS: 6,
            },
        )

    def test_group_identity_inverse_and_associativity_hold_exhaustively(self) -> None:
        identity = identity_permutation()
        actions = all_s4_permutations()
        for action in actions:
            inverse = inverse_permutation(action)
            self.assertEqual(compose_permutations(identity, action), action)
            self.assertEqual(compose_permutations(action, identity), action)
            self.assertEqual(compose_permutations(action, inverse), identity)
            self.assertEqual(compose_permutations(inverse, action), identity)
        for first in actions:
            for second in actions:
                for third in actions:
                    self.assertEqual(
                        compose_permutations(
                            compose_permutations(first, second), third
                        ),
                        compose_permutations(
                            first, compose_permutations(second, third)
                        ),
                    )

    def test_composition_matches_temporal_application_convention(self) -> None:
        values = ("a", "b", "c", "d")
        for first in all_s4_permutations():
            for second in all_s4_permutations():
                sequential = apply_permutation(
                    apply_permutation(values, first), second
                )
                composed = apply_permutation(
                    values, compose_permutations(first, second)
                )
                self.assertEqual(sequential, composed)

    def test_every_action_has_a_deterministic_minimal_transposition_rollout(self) -> None:
        expected_lengths = {
            IDENTITY_CLASS: 0,
            TRANSPOSITION_CLASS: 1,
            DOUBLE_TRANSPOSITION_CLASS: 2,
            THREE_CYCLE_CLASS: 2,
            FOUR_CYCLE_CLASS: 3,
        }
        generators = set(transposition_generators())
        self.assertEqual(len(generators), 6)
        for action in all_s4_permutations():
            first = decompose_into_transpositions(action)
            second = decompose_into_transpositions(action)
            self.assertEqual(first, second)
            self.assertEqual(compose_rollout(first), action)
            self.assertEqual(len(first), expected_lengths[permutation_class(action)])
            self.assertTrue(set(first).issubset(generators))

    def test_validation_rejects_nonpermutations_and_invalid_transpositions(self) -> None:
        for invalid in ((0, 1, 2), (0, 1, 1, 3), (0, 1, 2, 4)):
            with self.assertRaises(ValueError):
                validate_permutation(invalid)
        with self.assertRaises(ValueError):
            transposition(1, 1)

    def test_episode_generation_is_deterministic_balanced_and_duplicate_free(self) -> None:
        kwargs = {
            "split": "train",
            "keys": tuple(f"key-{index}" for index in range(12)),
            "values": tuple(f"value-{index}" for index in range(12)),
            "count": 32,
            "seed": 487,
        }
        first = generate_binding_algebra_episodes(**kwargs)
        second = generate_binding_algebra_episodes(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len({episode.episode_id for episode in first}), 32)
        self.assertEqual(
            [sum(episode.query_index == index for episode in first) for index in range(4)],
            [8, 8, 8, 8],
        )

    def test_registered_token_pools_are_globally_disjoint(self) -> None:
        assert_globally_disjoint_token_pools(self.config["token_pools"])
        changed = copy.deepcopy(self.config["token_pools"])
        changed["values"]["test"][0] = changed["keys"]["train"][0]
        with self.assertRaisesRegex(ValueError, "appears in"):
            assert_globally_disjoint_token_pools(changed)

    def test_action_class_partition_is_primitive_vs_composed(self) -> None:
        partition = self.config["action_partition"]
        assert_action_class_partition(
            partition["train_classes"], partition["held_out_classes"]
        )
        self.assertEqual(partition["train_classes"], [TRANSPOSITION_CLASS])
        self.assertEqual(
            set(partition["held_out_classes"]), set(HELD_OUT_COMPOSITION_CLASSES)
        )
        with self.assertRaisesRegex(ValueError, "leak"):
            assert_action_class_partition(
                [TRANSPOSITION_CLASS],
                [TRANSPOSITION_CLASS, THREE_CYCLE_CLASS],
            )

    def test_config_materialization_preserves_independent_episode_units(self) -> None:
        episodes = binding_algebra_episodes_from_config(self.config)
        counts = Counter(episode.split for episode in episodes)
        self.assertEqual(
            counts,
            {
                "calibration": 16,
                "train": 192,
                "validation": 64,
                "test": 64,
                "paraphrase": 64,
            },
        )
        test = [episode for episode in episodes if episode.split == "test"]
        paraphrase = [episode for episode in episodes if episode.split == "paraphrase"]
        for source, shifted in zip(test, paraphrase, strict=True):
            self.assertEqual(source.keys, shifted.keys)
            self.assertEqual(source.base_values, shifted.base_values)
            self.assertEqual(source.query_index, shifted.query_index)
            self.assertNotEqual(source.template, shifted.template)

    def test_cases_use_only_train_transpositions_and_held_out_cycles(self) -> None:
        cases = binding_algebra_cases_from_config(self.config)
        counts = Counter(case.split for case in cases)
        self.assertEqual(counts["calibration"], 16 * 3)
        self.assertEqual(counts["train"], 192 * 3)
        self.assertEqual(counts["validation"], 64 * 15)
        self.assertEqual(counts["test"], 64 * 15)
        self.assertEqual(counts["paraphrase"], 64 * 15)
        train_classes = {
            case.permutation_class for case in cases if case.split == "train"
        }
        protected_classes = {
            case.permutation_class
            for case in cases
            if case.split in {"test", "paraphrase"}
        }
        self.assertEqual(train_classes, {TRANSPOSITION_CLASS})
        self.assertEqual(protected_classes, set(HELD_OUT_COMPOSITION_CLASSES))
        self.assertTrue(
            all(
                permutation_changes_slot(case.target_permutation, case.query_index)
                for case in cases
            )
        )

    def test_query_changing_rosters_have_registered_per_class_counts(self) -> None:
        held_out = permutations_in_classes(HELD_OUT_COMPOSITION_CLASSES)
        for query_index in range(4):
            changing = [
                action
                for action in held_out
                if permutation_changes_slot(action, query_index)
            ]
            self.assertEqual(
                Counter(permutation_class(action) for action in changing),
                {
                    DOUBLE_TRANSPOSITION_CLASS: 3,
                    THREE_CYCLE_CLASS: 6,
                    FOUR_CYCLE_CLASS: 6,
                },
            )

    def test_test_and_paraphrase_cases_are_exact_action_pairs(self) -> None:
        cases = binding_algebra_cases_from_config(self.config)
        test = [case for case in cases if case.split == "test"]
        paraphrase = [case for case in cases if case.split == "paraphrase"]
        self.assertEqual(len(test), len(paraphrase))
        for source, shifted in zip(test, paraphrase, strict=True):
            self.assertEqual(source.query_index, shifted.query_index)
            self.assertEqual(source.target_permutation, shifted.target_permutation)
            self.assertEqual(source.generator_rollout, shifted.generator_rollout)

    def test_protocol_digest_is_deterministic_and_binds_actions(self) -> None:
        episodes = binding_algebra_episodes_from_config(self.config)
        cases = binding_algebra_cases_from_config(self.config)
        first = binding_algebra_protocol_digest(episodes, cases)
        second = binding_algebra_protocol_digest(episodes, cases)
        self.assertEqual(first, second)
        changed = list(cases)
        original = changed[0]
        alternative = transposition(0, 2)
        if not permutation_changes_slot(alternative, original.query_index):
            alternative = transposition(original.query_index, (original.query_index + 1) % 4)
        changed[0] = BindingAlgebraCase(
            case_id=original.case_id,
            split=original.split,
            episode_id=original.episode_id,
            query_index=original.query_index,
            target_permutation=alternative,
            permutation_class=TRANSPOSITION_CLASS,
            generator_rollout=(alternative,),
        )
        self.assertNotEqual(first, binding_algebra_protocol_digest(episodes, changed))

    def test_config_forbids_old_full_state_and_identity_shortcuts(self) -> None:
        self.assertFalse(self.config["execution_authorized"])
        self.assertEqual(
            self.config["capture"]["state_target"], "treated_minus_clean_fp32"
        )
        self.assertTrue(
            self.config["capture"]["full_residual_state_target_forbidden"]
        )
        self.assertTrue(
            self.config["capture"]["donor_activation_as_predictor_input_forbidden"]
        )
        self.assertTrue(self.config["meta_model"]["token_ids_as_features_forbidden"])
        self.assertTrue(
            self.config["meta_model"]["train_on_composed_targets_forbidden"]
        )
        self.assertEqual(
            self.config["baselines"]["full_quadratic_cross_terms"], "included"
        )
        self.assertIn(
            "oracle_relinearized_sequential_jvp",
            self.config["baselines"]["methods"],
        )


if __name__ == "__main__":
    unittest.main()
