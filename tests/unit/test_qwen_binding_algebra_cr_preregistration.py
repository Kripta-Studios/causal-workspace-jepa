from __future__ import annotations

import copy
import hashlib
import json
import re
import unittest
from pathlib import Path
from typing import Any

from causal_workspace_jepa.common.config import load_config


CONFIG = Path("configs/experiments/qwen_binding_algebra_cr_v1.yaml")
PARENT_CONFIG = Path("configs/experiments/qwen_binding_algebra_v2.yaml")


def _canonical_digest(config: dict[str, Any]) -> str:
    payload = copy.deepcopy(config)
    payload.pop("canonical_digest", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class QwenBindingAlgebraCRPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG)
        cls.parent = load_config(PARENT_CONFIG)

    def test_id_parent_and_digest_are_distinct_and_deterministic(self) -> None:
        config = self.config
        self.assertEqual(config["id"], "QWEN-BINDING-ALGEBRA-CR-V1")
        self.assertEqual(config["experiment_id"], config["id"])
        self.assertNotEqual(config["id"], self.parent["id"])
        linkage = config["parent_linkage"]
        self.assertEqual(
            linkage["parent_experiment_id"], "LLM-QWEN-BINDING-ALGEBRA-002"
        )
        self.assertEqual(
            linkage["parent_config"], "configs/experiments/qwen_binding_algebra_v2.yaml"
        )
        self.assertEqual(
            linkage["parent_config_sha256"],
            hashlib.sha256(PARENT_CONFIG.read_bytes()).hexdigest(),
        )
        parent_semantic = hashlib.sha256(
            json.dumps(
                self.parent,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(linkage["parent_semantic_sha256"], parent_semantic)
        self.assertTrue(linkage["exact_parent_required"])
        self.assertFalse(linkage["parent_config_may_be_modified"])
        self.assertTrue(linkage["authorization_inheritance"] == "forbidden")
        expected = _canonical_digest(config)
        self.assertEqual(config["canonical_digest"], expected)
        self.assertEqual(expected, _canonical_digest(copy.deepcopy(config)))
        contract = config["canonical_digest_contract"]
        self.assertEqual(contract["algorithm"], "sha256")
        self.assertEqual(contract["serialization"], "json_sort_keys_utf8_compact")
        self.assertEqual(contract["scope"], "parsed_config_without_canonical_digest_field")
        self.assertFalse(contract["includes_model_or_tokenizer_state"])
        self.assertFalse(contract["includes_outcomes_or_protected_artifacts"])

    def test_frozen_model_seeds_splits_and_thresholds_are_explicit_v2_values(self) -> None:
        config, parent = self.config, self.parent
        for key in (
            "model",
            "revision",
            "device",
            "dtype",
            "capture_activation_dtype",
            "attn_implementation",
            "local_files_only",
            "max_sequence_length",
            "seed",
        ):
            self.assertEqual(config[key], parent[key], key)
        self.assertEqual(config["splits"], parent["splits"])
        self.assertEqual(config["token_pools"], parent["token_pools"])
        self.assertEqual(config["meta_model"]["training_seeds"], [541, 547, 557])
        self.assertEqual(config["meta_model"]["training_seeds"], parent["meta_model"]["training_seeds"])
        frozen_gate_keys = (
            "clean_accuracy_min_each_split",
            "direct_permutation_accuracy_min_each_split",
            "exact_layer0_replay_atol",
            "composition_interaction_power_min",
            "best_quadratic_normalized_mse_min",
            "meta_model_mse_ratio_to_best_baseline_max",
            "behavior_agreement_margin_min",
            "paired_bootstrap_advantage_lower_min",
            "predicted_state_patch_recovery_min",
            "predicted_state_control_margin_min",
            "inverse_restoration_min",
            "seeds_required",
            "protected_splits_required",
            "bootstrap_draws",
            "bootstrap_seed",
        )
        for key in frozen_gate_keys:
            self.assertEqual(config["gates"][key], parent["gates"][key], key)
        self.assertEqual(config["gates"]["observed_layer21_patch_replay_atol"], 1.0e-6)

    def test_residual_only_target_and_baseline_classes_are_well_formed(self) -> None:
        config = self.config
        residual = config["residual_target"]
        self.assertEqual(residual["primary"], "residual_endpoint_or_trajectory")
        self.assertTrue(residual["complete_state_primary"] == "forbidden")
        self.assertTrue(residual["complete_delta_primary"] == "forbidden")
        self.assertEqual(residual["complete_state_control"], "registered_fair_comparator_only")
        self.assertEqual(residual["complete_delta_control"], "registered_fair_comparator_only")
        families = config["baseline_families"]
        deployable = set(families["deployable_residualizers"])
        fair = set(families["fair_comparators"])
        oracle = set(families["oracle_ceilings"])
        self.assertTrue(deployable)
        self.assertTrue(deployable.issubset(fair))
        self.assertTrue(oracle.isdisjoint(deployable))
        self.assertTrue(oracle.isdisjoint(fair))
        selection = config["baseline_selection"]
        self.assertEqual(selection["eligible_family"], "deployable_residualizers")
        self.assertEqual(set(selection["eligible_methods"]), deployable)
        self.assertEqual(
            selection["selection_split"], "validation"
        )
        self.assertEqual(selection["fit_splits"], ["train"])
        self.assertEqual(selection["metric"], "normalized_mse")
        self.assertEqual(
            selection["aggregation"], "ratio_of_sums_over_complete_base_episodes_before_division"
        )
        self.assertIsNone(selection["selected_method"])
        self.assertTrue(selection["selection_record_outcome_blind"])
        self.assertTrue(selection["selection_record_self_hash"] == "required_before_residual_fit")
        self.assertFalse(selection["protected_results_may_select_baseline"])

    def test_observed_prefix_oracle_cannot_define_baseline_star(self) -> None:
        config = self.config
        selection = config["baseline_selection"]
        oracle = set(config["baseline_families"]["oracle_ceilings"])
        self.assertTrue(oracle.issubset(set(selection["excluded_oracle_ceilings"])))
        self.assertTrue(
            oracle.isdisjoint(set(selection["eligible_methods"]))
        )
        baselines = config["baselines"]
        self.assertFalse(baselines["observed_prefix_oracle_selects_baseline_star"])
        self.assertFalse(baselines["observed_prefix_oracle_defines_residual_target"])
        self.assertFalse(baselines["oracle_inputs_count_as_deployment_inputs"])
        self.assertTrue(baselines["observed_prefix_oracle_in_leaderboard"])

    def test_all_phase_locks_are_ordered_and_protected_phase_is_closed(self) -> None:
        config = self.config
        locks = config["phase_locks"]
        self.assertEqual(set(locks), {"B0", "B1", "B2", "B3", "B4"})
        self.assertEqual([locks[name]["order"] for name in ("B0", "B1", "B2", "B3", "B4")], [0, 1, 2, 3, 4])
        self.assertEqual(locks["B0"]["allowed_splits"], ["calibration", "train", "validation"])
        self.assertEqual(locks["B1"]["decision_split"], "validation")
        self.assertEqual(locks["B2"]["allowed_splits"], ["train", "validation"])
        self.assertEqual(locks["B3"]["training_seeds"], [541, 547, 557])
        self.assertEqual(locks["B4"]["allowed_splits"], ["test", "paraphrase"])
        for name in ("B0", "B1", "B2", "B3", "B4"):
            self.assertTrue(locks[name]["requires_execution_authorization"], name)
            self.assertTrue(locks[name]["schema_only_dry_run_without_authorization"], name)
            self.assertFalse(locks[name]["qwen_execution_without_authorization"], name)
        self.assertTrue(locks["B4"]["requires_clean_pushed_commit"])
        self.assertTrue(locks["B4"]["requires_self_hashed_plan"])
        self.assertFalse(locks["B4"]["protected_results_select_methods"])
        self.assertEqual(
            locks["B4"]["interpretability_comparisons"],
            ["sae", "qwen_scope", "transcoder", "atp_star", "jacobian_rankings"],
        )
        self.assertTrue(locks["B4"]["feature_claim_requires_direct_ablation_or_patch"])
        access = config["phase_policy"]["access"]
        self.assertNotEqual(access["phase_0"]["output_root"], access["phase_1_train"]["output_root"])
        self.assertNotEqual(access["phase_1_train"]["output_root"], access["protected_eval"]["output_root"])
        self.assertFalse(config["phase_policy"]["protected_results_select_methods"])
        self.assertFalse(config["phase_policy"]["protected_results_select_thresholds"])
        self.assertFalse(config["phase_policy"]["protected_results_select_primary"])

    def test_direct_replay_contract_is_complete_and_fail_closed(self) -> None:
        replay = self.config["direct_replay"]
        self.assertTrue(replay["observed_target_replay_required_first"])
        self.assertEqual(replay["observed_target_atol"], 1.0e-6)
        self.assertEqual(replay["replacement_mode"], "complete_state_at_every_registered_site")
        for key in (
            "partial_state_replacement_forbidden",
            "missing_site_fails_closed",
            "nonfinite_value_fails_closed",
            "exact_treated_prefix_as_predictor_input_forbidden",
            "same_model_revision_required",
            "same_runtime_required",
            "same_precision_required",
            "same_tokenizer_and_tokenization_required",
            "same_rng_state_required",
            "same_decoding_contract_required",
            "failure_is_not_zero_effect",
            "direct_replay_required_for_causal_claim",
        ):
            self.assertTrue(replay[key], key)
        self.assertIn("activation_delta", replay["endpoints"])
        self.assertIn("candidate_behavior", replay["endpoints"])

    def test_b2_target_sufficiency_and_hard_negative_contracts_are_frozen(self) -> None:
        sufficiency = self.config["target_sufficiency"]
        self.assertTrue(sufficiency["exact_target_embedding_oracle"] == "required")
        self.assertEqual(sufficiency["oracle_fit_splits"], "train_only")
        self.assertTrue(sufficiency["oracle_direct_replay_required"])
        self.assertTrue(sufficiency["oracle_candidate_behavior_required"])
        self.assertTrue(sufficiency["train_only_pca_ridge_comparator"] == "required")
        self.assertTrue(sufficiency["oracle_must_beat_train_only_pca_ridge"])
        self.assertEqual(
            set(sufficiency["comparison_endpoints"]),
            {"activation_delta", "role_logit_delta", "candidate_behavior"},
        )
        for key in (
            "donor_dominance_rejects_target",
            "prompt_dominance_rejects_target",
            "position_dominance_rejects_target",
            "token_id_dominance_rejects_target",
        ):
            self.assertTrue(sufficiency[key], key)
        hard_negatives = self.config["hard_negatives"]
        self.assertTrue(hard_negatives["mandatory"])
        self.assertIn("same_context_wrong_action", hard_negatives["roster"])
        self.assertIn("same_context_wrong_prefix", hard_negatives["roster"])
        self.assertIn("covariance_matched_delta", hard_negatives["roster"])
        self.assertEqual(hard_negatives["clustered_inference_unit"], "complete_base_episode")

    def test_within_context_roster_and_visreg_retention_are_direct_replay_only(self) -> None:
        comparison = self.config["within_context_intervention_comparisons"]
        self.assertEqual(comparison["target"], "residual_regression_only")
        self.assertEqual(
            comparison["candidates"],
            [
                "residual_regression_only",
                "residual_ijepa",
                "residual_ijepa_within_context_nce",
                "residual_ijepa_residual_visreg",
                "residual_ijepa_within_context_nce_and_residual_visreg",
            ],
        )
        self.assertEqual(
            comparison["target_encoder_variants"],
            ["end_to_end_target_encoder", "stop_gradient_target_encoder", "ema_stop_gradient_target_encoder"],
        )
        for factor in (
            "prompt",
            "recipient",
            "answer_format",
            "token_position",
            "layer_family",
            "patch_type",
            "norm",
            "covariance",
        ):
            self.assertIn(factor, comparison["candidate_sets_hold_constant_where_possible"])
        self.assertTrue(comparison["factor_matching"]["matching_is_frozen_before_validation"])
        self.assertTrue(comparison["intervention_classification_or_retrieval_is_not_selection_endpoint"])
        hard_negatives = set(self.config["hard_negatives"]["roster"])
        for name in (
            "same_donor_different_site",
            "different_donor_norm_cov_matched",
            "similar_state_error_different_behavior",
            "similar_behavior_different_residual",
            "identity_noop",
            "inverse_action",
            "commuting_composition",
            "noncommuting_composition",
        ):
            self.assertIn(name, hard_negatives, name)
        visreg = self.config["visreg_contract"]
        self.assertEqual(visreg["applies_to"], ["normalized_residual", "normalized_residual_trajectory"])
        self.assertFalse(visreg["applies_to_complete_state"])
        self.assertFalse(visreg["applies_to_complete_delta"])
        self.assertTrue(visreg["primary_complete_state_visreg_forbidden"])
        retention = self.config["candidate_retention"]
        self.assertEqual(retention["selection_split"], "validation")
        self.assertTrue(retention["direct_replay_required"])
        self.assertTrue(retention["direct_behavior_required"])
        self.assertTrue(retention["intervention_classification_is_insufficient"])
        self.assertTrue(retention["intervention_retrieval_is_insufficient"])

    def test_loss_comparison_and_nuisance_rosters_cannot_select_after_protected_access(self) -> None:
        loss = self.config["loss"]
        for key in (
            "replay_endpoint_weight",
            "replay_kl_weight",
            "inverse_restoration_weight",
            "sequential_composition_weight",
            "commutator_interaction_weight",
            "matched_control_specificity_weight",
            "uncertainty_calibration_weight",
            "nuisance_anti_shortcut_weight",
        ):
            self.assertIn(key, loss)
        terms = set(self.config["loss_roster"]["terms"])
        self.assertTrue({"replay_endpoint", "replay_kl", "inverse_restoration"}.issubset(terms))
        self.assertTrue({"sequential_composition", "commutator_noncommuting_interaction"}.issubset(terms))
        self.assertTrue({"matched_control_specificity", "uncertainty_calibration", "nuisance_anti_shortcut"}.issubset(terms))
        self.assertEqual(
            self.config["target_encoder_comparisons"]["candidates"],
            ["end_to_end_target_encoder", "stop_gradient_target_encoder", "ema_stop_gradient_target_encoder"],
        )
        nuisance = self.config["ablations"]["nuisance"]
        for factor in ("episode_id", "prompt_id", "slow_timestamp", "persistent_marker", "predictable_bit_string", "norm_matched_delta", "covariance_matched_delta"):
            self.assertIn(factor, nuisance["planted_suite"], factor)
        self.assertTrue(nuisance["dominance_rejects_target"])
        decision = self.config["decision_logic"]
        self.assertEqual(decision["seed_count"], 3)
        self.assertEqual(decision["seed_passes_required"], 2)
        self.assertTrue(decision["subgroup_results_replace_conjunctive_decision"] is False)
        self.assertEqual(decision["bootstrap"]["unit"], "complete_base_episode")
        self.assertTrue(decision["bootstrap"]["action_expanded_rows_are_not_independent"])

    def test_preregistration_is_outcome_blind_unauthorized_and_cannot_promote(self) -> None:
        config = self.config
        self.assertFalse(config["execution_authorized"])
        self.assertEqual(config["status"], "PREREGISTERED_OUTCOME_BLIND_NOT_AUTHORIZED")
        authorization = config["authorization"]
        for key in (
            "execution_authorized",
            "protected_execution_authorized",
            "protected_rows_closed",
            "tokenizer_access_before_authorization",
            "qwen_model_access_before_authorization",
            "protected_activation_read_before_authorization",
            "protected_result_read_before_authorization",
            "protected_plan_commit_required",
            "clean_git_required",
            "source_commit_must_be_ancestor",
            "authorization_change_requires_new_commit",
            "authorization_change_requires_independent_review",
            "fail_closed_on_missing_flag",
            "fail_closed_on_true_flag_without_plan",
            "fail_closed_on_parent_authorization",
        ):
            expected = False if key.endswith("_before_authorization") else True
            if key in {"execution_authorized", "protected_execution_authorized"}:
                expected = False
            self.assertEqual(authorization[key], expected, key)
        primary = config["protected_primary"]
        self.assertTrue(primary["exactly_one_protected_primary"])
        self.assertIsNone(primary["selected_primary"])
        self.assertEqual(primary["selection_status"], "pending_validation")
        for key in (
            "protected_results_may_select_primary",
            "protected_results_may_replace_primary",
            "protected_results_may_promote_secondary",
            "protected_results_may_change_hyperparameters",
        ):
            self.assertFalse(primary[key], key)
        text = CONFIG.read_text(encoding="utf-8")
        self.assertNotRegex(text, re.compile(r"(?i)observed_(?:test|paraphrase)_(?:mse|accuracy|effect|result)"))
        self.assertNotIn("protected_outcome", text.lower())

    def test_comparators_and_ablation_roster_are_registered_without_winner(self) -> None:
        config = self.config
        strongest = config["strongest_comparisons"]
        self.assertIn("full_quadratic_hvp", strongest["analytical"])
        self.assertIn("exact_local_jvp", strongest["analytical"])
        self.assertIn("capacity_matched_mlp", strongest["learned"])
        self.assertIn("complete_delta_control", strongest["learned"])
        self.assertIn("complete_state_control", strongest["learned"])
        self.assertTrue(strongest["protected_primary_must_beat_strongest_analytical"])
        self.assertTrue(strongest["protected_primary_must_beat_strongest_learned"])
        self.assertTrue(strongest["oracle_is_headroom_only"])
        ablations = config["ablations"]
        self.assertTrue(ablations["nuisance"]["registered"])
        self.assertTrue(ablations["nuisance"]["dominance_rejects_target"])
        self.assertFalse(ablations["tcr"]["default"] == "enabled")
        self.assertFalse(ablations["vis"]["default"] == "enabled")
        self.assertEqual(ablations["tcr_vis"]["tcr_enabled"], False)
        self.assertEqual(ablations["tcr_vis"]["visreg_enabled"], False)
        self.assertTrue(ablations["router"]["values_retain_source_dimension"])
        self.assertEqual(
            set(ablations["router"]["independent_ablations"]),
            {"routing_keys_removed", "transported_values_removed", "keys_only", "values_only"},
        )


if __name__ == "__main__":
    unittest.main()
