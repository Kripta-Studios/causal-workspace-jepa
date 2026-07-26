from __future__ import annotations

import copy
import unittest
from collections import Counter

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_protocol import (
    IDENTITY_CONTROL,
    INVERSE_CONTROL,
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
    assert_phase_split_access,
    assert_globally_disjoint_token_pools,
    binding_algebra_cases_from_config,
    binding_algebra_controls,
    binding_algebra_episodes_from_config,
    binding_algebra_protocol_digest,
    compose_permutations,
    compose_rollout,
    decompose_into_transpositions,
    generate_binding_algebra_episodes,
    identity_permutation,
    inverse_permutation,
    permutation_matrix,
    permutation_changes_slot,
    permutation_class,
    permutations_in_classes,
    phase_access_contract,
    predictor_generator_action_matrices,
    rollout_prefixes,
    transposition,
    transposition_generators,
    validate_permutation,
)


class QwenBindingAlgebraProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("configs/experiments/qwen_binding_algebra_v2.yaml")

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

    def test_permutation_matrices_match_actions_and_all_compositions(self) -> None:
        values = np.arange(4, dtype=np.int64)
        actions = all_s4_permutations()
        for action in actions:
            matrix = permutation_matrix(action)
            self.assertEqual(matrix.shape, (4, 4))
            np.testing.assert_array_equal(
                matrix @ values,
                np.asarray(apply_permutation(values, action)),
            )
            np.testing.assert_array_equal(matrix.sum(axis=0), np.ones(4))
            np.testing.assert_array_equal(matrix.sum(axis=1), np.ones(4))
        for first in actions:
            for second in actions:
                np.testing.assert_array_equal(
                    permutation_matrix(compose_permutations(first, second)),
                    permutation_matrix(second) @ permutation_matrix(first),
                )

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
            prefixes = rollout_prefixes(first)
            self.assertEqual(prefixes[0], identity_permutation())
            self.assertEqual(prefixes[-1], action)
            self.assertEqual(len(prefixes), len(first) + 1)

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

    def test_identity_inverse_controls_are_explicit_and_restore_exactly(self) -> None:
        episodes = binding_algebra_episodes_from_config(self.config)
        cases = binding_algebra_cases_from_config(self.config)
        controls = binding_algebra_controls(episodes, cases)
        counts = Counter(control.control_type for control in controls)
        self.assertEqual(counts[IDENTITY_CONTROL], len(episodes))
        self.assertEqual(counts[INVERSE_CONTROL], len(cases))
        self.assertEqual(len({control.control_id for control in controls}), len(controls))
        for control in controls:
            self.assertEqual(
                compose_rollout(control.action_rollout),
                control.expected_permutation,
            )
            self.assertEqual(control.expected_permutation, identity_permutation())
            self.assertEqual(
                rollout_prefixes(control.action_rollout)[-1],
                identity_permutation(),
            )

    def test_predictor_receives_generator_program_not_composed_target_matrix(self) -> None:
        cases = binding_algebra_cases_from_config(self.config)
        for case in cases:
            matrices = predictor_generator_action_matrices(case)
            self.assertEqual(len(matrices), len(case.generator_rollout))
            for matrix, generator in zip(
                matrices, case.generator_rollout, strict=True
            ):
                np.testing.assert_array_equal(matrix, permutation_matrix(generator))
            if case.permutation_class in HELD_OUT_COMPOSITION_CLASSES:
                target = permutation_matrix(case.target_permutation)
                self.assertTrue(
                    all(not np.array_equal(matrix, target) for matrix in matrices)
                )

    def test_phase_policy_blocks_protected_rows_before_frozen_phase(self) -> None:
        contract = phase_access_contract(self.config)
        self.assertEqual(
            contract["allowed_splits"]["phase_0"],
            ("calibration", "train", "validation"),
        )
        self.assertEqual(len(set(contract["output_roots"].values())), 3)
        self.assertEqual(
            assert_phase_split_access(
                self.config, "phase_0", ("calibration", "validation")
            ),
            ("calibration", "validation"),
        )
        with self.assertRaisesRegex(RuntimeError, "BLOCKED_SPLIT_ACCESS"):
            assert_phase_split_access(self.config, "phase_0", ("validation", "test"))
        with self.assertRaisesRegex(RuntimeError, "BLOCKED_SPLIT_ACCESS"):
            assert_phase_split_access(
                self.config, "phase_1_train", ("train", "paraphrase")
            )

    def test_protocol_digest_is_deterministic_and_binds_actions(self) -> None:
        episodes = binding_algebra_episodes_from_config(self.config)
        cases = binding_algebra_cases_from_config(self.config)
        controls = binding_algebra_controls(episodes, cases)
        first = binding_algebra_protocol_digest(
            self.config, episodes, cases, controls
        )
        second = binding_algebra_protocol_digest(
            self.config, episodes, cases, controls
        )
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
        self.assertNotEqual(
            first,
            binding_algebra_protocol_digest(
                self.config, episodes, changed, controls
            ),
        )

        changed_controls = list(controls)
        changed_control = changed_controls[-1]
        changed_controls[-1] = type(changed_control)(
            control_id=f"{changed_control.control_id}:changed",
            split=changed_control.split,
            episode_id=changed_control.episode_id,
            control_type=changed_control.control_type,
            parent_case_id=changed_control.parent_case_id,
            action_rollout=changed_control.action_rollout,
            expected_permutation=changed_control.expected_permutation,
        )
        self.assertNotEqual(
            first,
            binding_algebra_protocol_digest(
                self.config, episodes, cases, changed_controls
            ),
        )

    def test_config_forbids_old_full_state_and_identity_shortcuts(self) -> None:
        self.assertFalse(self.config["execution_authorized"])
        self.assertEqual(
            self.config["capture"]["state_target"],
            "cumulative_prefix_treated_minus_same_episode_clean_original_fp32",
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
        self.assertTrue(
            self.config["meta_model"]["composed_target_matrix_as_input_forbidden"]
        )
        self.assertEqual(
            self.config["baselines"]["full_quadratic_cross_terms"], "included"
        )
        self.assertIn(
            "oracle_relinearized_sequential_jvp",
            self.config["baselines"]["methods"],
        )
        self.assertIn("affine_generator_rollout", self.config["baselines"]["methods"])
        self.assertIn("s4_equivariant_linear", self.config["baselines"]["methods"])
        self.assertEqual(
            self.config["baselines"]["direct_primitive_addition_semantics"],
            "each_generator_effect_executed_from_same_clean_origin_then_summed",
        )


if __name__ == "__main__":
    unittest.main()
