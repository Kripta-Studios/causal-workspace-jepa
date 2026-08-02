from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from causal_workspace_jepa.common.config import load_config


CONFIG_PATH = Path("configs/experiments/cr_ac_jepa_v1.yaml")
OFFICIAL_COMPETENCE_PATH = Path(
    "configs/experiments/eb_jepa_two_rooms_competence.yaml"
)


def _config() -> dict[str, Any]:
    return load_config(CONFIG_PATH)


def _canonical_digest(config: dict[str, Any]) -> str:
    payload = copy.deepcopy(config)
    payload["canonical_digest"].pop("value", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase_allowed(
    config: dict[str, Any],
    phase: str,
    passed_gates: set[str],
    *,
    authorized: bool,
) -> bool:
    """Reference fail-closed phase guard used to lock the YAML contract."""

    policy = config["phase_policy"]
    required = set(policy[f"{phase}_requires"])
    if not required.issubset(passed_gates):
        return False
    if phase == "protected":
        return authorized and config["execution_authorized"]
    return authorized and config["execution_authorized"]


def test_config_loads_with_distinct_id_and_stable_canonical_digest() -> None:
    config = _config()
    assert config["id"] == "CR-AC-JEPA-001"
    assert config["protocol"] == "CR-AC-JEPA-V1"
    assert config["track"] == "A"
    assert config["status"] == "PREREGISTERED_NOT_AUTHORIZED"
    learning = config["learning_contract"]
    assert learning["paradigm"] == "reward-free action-supervised learning from offline trajectories"
    assert learning["recorded_actions"] is True
    assert learning["semantic_labels"] is False
    assert learning["reward_annotations"] is False
    assert learning["reinforcement_learning"] == "forbidden"
    digest = config["canonical_digest"]
    assert digest["algorithm"] == "sha256"
    assert digest["serialization"] == "canonical_json_sorted_keys_compact"
    assert digest["exclude_paths"] == ["canonical_digest.value"]
    assert len(digest["value"]) == 64
    assert digest["value"] == _canonical_digest(config)
    assert digest["value"] == _canonical_digest(_config())


def test_authorization_and_provenance_are_protected_by_default() -> None:
    config = _config()
    assert config["execution_authorized"] is False
    authorization = config["authorization"]
    assert authorization["execution_authorized"] is False
    assert authorization["protected"] is False
    assert authorization["expensive"] is False
    assert authorization["fail_closed"] is True
    assert authorization["required_authorization_commit_and_push"] is True
    assert config["provenance"]["require_clean_worktree"] is True
    assert config["provenance"]["record_canonical_digest"] is True


def test_stage_zero_freezes_observable_residuals_before_any_learning() -> None:
    config = _config()
    stage = config["stage_0"]
    assert stage["execution_authorized"] is True
    assert stage["protected"] is False
    assert stage["deterministic"] is True
    assert stage["state_source"] == "observable_ground_truth_physical_state"
    assert stage["frozen_encoder"] is True
    assert stage["target_is_observable_or_frozen_encoder"] is True
    assert stage["learning_allowed"] == "conditional_after_finite_residual_floor"
    assert stage["no_learning_before_floors"] is True
    assert stage["cartesian_factorial_expansion"] == "forbidden"
    assert stage["systems"] == [
        "linear",
        "quadratic",
        "nonlinear_compositional",
        "predictable_nuisance",
    ]
    floor = stage["finite_residual"]
    assert floor["minimum_relative_energy"] > 0.0
    assert floor["minimum_absolute_energy"] > 0.0
    assert floor["minimum_repeatability"] >= 0.99
    assert floor["required_repetitions"] >= 3
    assert floor["gate_before_residual_learning"] is True
    frozen = stage["frozen_targets"]
    assert frozen["baseline_star"] == "selected_before_residual_model_fit"
    assert frozen["normalization"] == "fit_train_only_then_frozen"
    assert frozen["drift_report_required"] is True
    assert frozen["replay_report_required"] is True
    assert stage["subphases"]["baseline_and_floor"]["learning_allowed"] is False
    assert stage["subphases"]["eligible_synthetic_residual_learning"]["learning_allowed"] is True
    assert (
        stage["subphases"]["eligible_synthetic_residual_learning"][
            "nonlinear_replay_improvement_required"
        ]
        is True
    )
    assert (
        stage["subphases"]["eligible_synthetic_residual_learning"][
            "linear_and_quadratic_zero_discovery_expected"
        ]
        is True
    )
    assert (
        stage["subphases"]["eligible_synthetic_residual_learning"]["requires"]
        == ["baseline_and_floor_pass", "train_validation_split", "frozen_target_statistics"]
    )


def test_a0_is_an_exact_lock_to_the_official_three_seed_competence_contract() -> None:
    config = _config()
    official = load_config(OFFICIAL_COMPETENCE_PATH)
    a0 = config["a0"]
    assert a0["config"] == str(OFFICIAL_COMPETENCE_PATH).replace("\\", "/")
    assert a0["experiment_id"] == official["id"]
    assert a0["source_revision"] == official["revision"]
    assert a0["seeds"] == official["training_seeds"]
    assert a0["planner_arms"] == official["planner_arms"]
    assert a0["mechanistic_arm"] == official["mechanistic_eligibility_arm"]
    assert a0["checkpoint_epochs"] == official["checkpoint_epochs"]
    assert a0["episodes_per_seed"] == official["num_episodes"]
    assert a0["environment_seed"] == official["environment_seed"]
    assert a0["action_max_norm"] == official["action_max_norm"]
    assert a0["overall_success_floor"] == official["overall_success_threshold"]
    assert a0["per_seed_success_floor"] == official["per_seed_success_threshold"]
    assert a0["zero_action_violations_required"] is True
    assert a0["mechanistic_lock"] == "no_mechanistic_phase_without_a0_pass"
    assert a0["execution_authorized"] is False
    assert a0["protected"] is False
    assert config["gates"]["A0"]["failure_status"] == "INELIGIBLE_TASK"


def test_a1_to_a3_and_funnel_freeze_the_single_validation_primary() -> None:
    config = _config()
    gates = config["gates"]
    assert set(gates) == {"A0", "A1", "A2", "A3"}
    assert gates["A1"]["name"] == "dense_controllable_representation"
    assert gates["A1"]["prerequisites"] == [
        "stage_0_complete",
        "finite_residual_floor",
        "repeatability_floor",
        "frozen_baseline_target",
    ]
    assert gates["A1"]["dense_tokens_required"] is True
    assert gates["A1"]["residual_stability_floor_included"] is True
    assert gates["A2"]["name"] == "exact_recurrent_and_path_interventions"
    assert gates["A2"]["prerequisites"] == ["a0_pass", "a1_pass"]
    assert gates["A2"]["exact_recurrent_intervention_required"] is True
    assert gates["A2"]["causal_path_patch_required"] is True
    assert gates["A2"]["validation_only_selection"] is True
    assert gates["A3"]["name"] == "planning_and_direct_causal_use"
    assert gates["A3"]["protected_primary_count"] == 1
    assert gates["A3"]["direct_replay_required"] is True
    assert gates["A3"]["matched_controls_required"] is True
    assert gates["A3"]["closed_loop_required"] is True
    funnel = config["funnel"]
    assert list(funnel)[:5] == ["stage_0", "stage_1", "stage_2", "stage_3", "stage_4"]
    assert funnel["stage_2"]["maximum_retained_per_track"] == 2
    assert funnel["stage_4"]["primary_count"] == 1
    assert funnel["stage_4"]["secondary_cannot_replace_primary"] is True
    assert funnel["validation_selection"]["primary_count"] == 1
    assert funnel["validation_selection"]["selection_before_protected"] is True
    assert funnel["validation_selection"]["protected_results_never_select_methods"] is True


def test_factorial_and_intact_grammar_are_exact_and_separately_registered() -> None:
    config = _config()
    factorial = config["factorial"]
    assert set(factorial["tokenization"]) == {"pooled", "dense_patch_object_tokens"}
    assert set(factorial["action_encoding"]) == {"vector", "masked_visual_action"}
    assert set(factorial["query_direction"]) == {"forward_only", "forward_inverse"}
    assert set(factorial["target_modes"]) == {"next", "delta", "residual"}
    assert "TCR-VIS" in factorial["required_axes"]
    assert factorial["dense_attention"] == "block_causal"
    assert factorial["capacity_matched"] is True
    assert factorial["data_scales"] == "fixed_three_scale_cells"
    assert factorial["data_scale_names"] == "small_medium_large"
    assert factorial["data_scale_selection"] == "fixed_validation_cells_only"
    assert factorial["cartesian_product"] == "forbidden"
    assert factorial["full_expensive_factorial_allowed"] is False
    actor = config["actor_factorial"]
    assert actor["required"] is True
    assert actor["cartesian_product"] == "forbidden"
    assert actor["cells"] == [
        "forward_only",
        "local_inverse_only",
        "goal_intent_only",
        "full_shared_local_goal",
        "independent_parameter_matched_actors",
        "independent_double_capacity_actors",
        "shared_trunk_separate_outputs",
        "without_z_times_m",
        "with_z_times_m",
        "TCR-VIS",
        "CR_residual_forward",
    ]
    assert actor["full_expensive_factorial_forbidden_until"] == [
        "stage_0_complete",
        "stage_1_complete",
    ]
    assert actor["full_expensive_factorial_allowed"] is False
    intact = config["intact"]
    grammar = intact["grammar"]
    assert grammar["slot_count"] == 4
    assert grammar["slots"] == [
        "z_t",
        "m_t",
        "z_t_times_m_t",
        "action_embedding_a_t_minus_1",
    ]
    assert grammar["exact_order"] is True
    assert grammar["local_intent"]["successor_attachment"] == "real_successor_attached"
    assert grammar["local_intent"]["successor_gradient"] == "enabled"
    assert grammar["goal_intent"]["goal_gradient"] == "stopped"
    assert grammar["goal_intent"]["current_gradient"] == "enabled"
    assert grammar["local_goal_pointwise_equality"] == "forbidden"
    assert intact["action_likelihood"]["separate_losses"] is True
    assert intact["action_likelihood"]["pointwise_equality_constraint"] is False


def test_direct_mask_and_actor_disabled_planner_controls_cannot_leak_future_state() -> None:
    config = _config()
    direct = config["direct"]
    assert direct["search"] == "zero_search"
    assert direct["candidate_sequences"] is False
    assert direct["terminal_latent_cost_calls"] is False
    assert direct["first_chunk_only_reencode"] is True
    assert direct["mode_selection"] == "frozen_validation_selected"
    assert direct["uncertainty_local_verify"] is True
    assert direct["recurrence"]["predicts_action_chunks_recurrently"] is True
    assert direct["recurrence"]["first_chunk_only_reencode"] is True
    assert direct["recurrence"]["reencode_after_executed_chunk"] is True
    assert direct["recurrence"]["uncertainty_local_verify"] is True
    assert direct["recurrence"]["mode_selection_uses_future_successors"] is False
    assert direct["recurrence"]["forward_predictor"] == "unchanged_registered_forward_predictor"
    assert direct["recurrence"]["each_selected_chunk_uses_forward_predictor"] is True
    assert direct["recurrence"]["execute_only_first_registered_chunk"] is True
    assert direct["recurrence"]["reencode_next_real_observation"] is True
    mask = direct["deployment_input_mask"]
    assert mask["exact"] is True
    assert "current_observation" in mask["allowed"]
    assert "frozen_goal_anchor" in mask["allowed"]
    assert "previous_executed_action_context" in mask["allowed"]
    forbidden = set(mask["forbidden"])
    assert {"future_successors", "future_trajectories", "rewards", "success_flags"}.issubset(
        forbidden
    )
    assert {"absolute_time", "relative_time", "episode_length", "policy_identity"}.issubset(
        forbidden
    )
    planners = config["planner_controls"]
    for name in ("actor_disabled_cem", "actor_disabled_mppi"):
        assert planners[name]["actor_disabled"] is True
        assert planners[name]["actor_gradients"] == "forbidden"
        assert planners[name]["actor_selects_candidates"] is False
    verifier = planners["centered_verifier"]
    assert verifier["enabled"] is True
    assert verifier["centered_on_direct_proposal"] is True
    assert verifier["search_scope"] == "local_only"
    assert verifier["broad_search"] == "forbidden"
    assert set(planners["action_heads"]) == {"gaussian", "mixture"}


def test_controls_resources_and_phase_guard_fail_closed() -> None:
    config = _config()
    controls = config["controls"]
    assert controls["action_quotient"]["required"] is True
    quotient = controls["action_quotient"]
    assert (
        quotient["equivalence_relation"]
        == "fixed_current_state_same_supported_conditional_expert_action_law"
    )
    assert quotient["fixed_current_state_only"] is True
    assert quotient["same_supported_conditional_expert_action_law"] is True
    assert quotient["metrics"] == [
        "predicted_to_expert_local_knn_overlap",
        "centered_kernel_or_linear_cka",
        "pointwise_action_r2",
        "effective_rank",
        "conditional_action_nll",
        "actor_disabled_planning",
        "direct_success",
        "optional_verified_success",
        "calibration",
    ]
    assert controls["support_generalization"]["held_out_action_support"] is True
    assert controls["gauge_pairing"]["dual_coordinate_transport"] == "required"
    assert controls["gauge_pairing"]["pooled_unpaired_overlap_is_null_control"] is True
    assert controls["gauge_pairing"]["controls"] == [
        "native",
        "identity_map",
        "calibrated",
        "shuffled",
        "same_objective_cross_seed",
        "reverse_pairings",
    ]
    calibration = controls["gauge_pairing"]["coordinate_calibration"]
    assert calibration["can_restore_existing_learned_conditional"] is True
    assert calibration["can_create_absent_action_dependence"] is False
    support = controls["support_generalization"]
    assert support["held_out_trajectory_families"] is True
    assert support["held_out_obstacles"] is True
    assert support["held_out_contact_regimes"] is True
    assert support["held_out_distractors"] is True
    assert support["held_out_dynamics"] is True
    assert support["action_equivalent_counterfactuals"] is True
    assert support["action_distinct_counterfactuals"] is True
    assert support["post_first_step_drift"] == "required"
    assert controls["nuisance"]["nuisance_dominance_rejects_target"] is True
    assert controls["nuisance"]["planted_suite"] == [
        "episode_id",
        "slow_timestamp",
        "persistent_camera_marker",
        "persistent_camera_template_marker",
        "predictable_bit_string",
        "norm_matched_direction",
        "covariance_matched_irrelevant_control",
    ]
    assert controls["nuisance"]["capacity_matching_required"] is True
    assert controls["nuisance"]["clock_drift_phase_episode_length_controls"] is True
    path = controls["causal_path_replay"]
    assert path["direct_observed_replay_before_predicted_replay"] is True
    assert path["predicted_state_complete_replacement"] is True
    assert path["necessity_sufficiency_faithfulness"] == "required"
    same = config["same_trajectory_nce"]
    assert same["next_latent_mse"] == "required"
    assert same["within_trajectory_nce_targets"] == ["full_future", "delta", "residual"]
    assert same["residual_visreg_variants"] == [
        "residual_without_visreg",
        "residual_with_visreg",
    ]
    assert same["predictor_training_modes"] == ["e2e", "stop_gradient", "ema"]
    assert same["full_episode_temporal_mean"] == "forbidden"
    assert same["temporal_exclusion"]["distance_weighting"] == "inverse_temporal_distance"
    sensitivity = same["action_sensitivity_gate"]
    assert sensitivity["shuffled_action_must_deteriorate"] is True
    assert sensitivity["sign_flipped_action_must_deteriorate"] is True
    assert sensitivity["norm_matched_action_must_deteriorate"] is True
    assert sensitivity["counterfactual_action_must_deteriorate"] is True
    decoder = config["action_decoder"]
    assert decoder["comparisons"] == [
        "displacement_decoder",
        "concatenated_endpoint_decoder",
    ]
    assert decoder["displacement_input"] == "z_t_plus_1_minus_z_t"
    assert decoder["endpoint_input"] == "concatenate_z_t_and_z_t_plus_1"
    sampler = config["adaptive_intervention_sampler"]
    assert sampler["status"] == "optional_train_only"
    assert sampler["allowed_splits"] == ["train"]
    assert sampler["forbidden_splits"] == ["validation", "protected_test"]
    assert sampler["validation_test_unbiased"] is True
    assert sampler["no_reinforcement_learning"] is True
    assert set(sampler["triggers"]) == {
        "nontrivial_residual",
        "uncertainty",
        "state_disagreement",
        "behavior_disagreement",
        "borderline_behavior",
        "compositions",
        "on_manifold_support",
    }
    assert set(sampler["downweight"]) == {"trivial", "linear", "duplicate", "off_manifold"}
    adoption = config["intact_adoption"]
    assert adoption["required_direct_endpoint"] == "held_out_direct_improvement"
    assert adoption["required_actor_disabled_criterion"] == "at_least_one_actor_disabled_criterion"
    assert adoption["required_controls"] == [
        "nuisance",
        "support",
        "multimodality",
        "pairing",
    ]
    assert adoption["all_required_controls_must_pass"] is True
    assert adoption["failure_label"] == "imitation_interface_only"
    resources = config["resource_preflight"]
    assert resources["required"] is True
    assert resources["fail_closed"] is True
    assert resources["blocked_status"] == "EXECUTION_BLOCKED"
    assert resources["unavailable_status"] == "EXECUTION_BLOCKED"
    policy = config["phase_policy"]
    assert policy["fail_closed"] is True
    assert policy["protected_rows_inaccessible_before_authorization"] is True
    assert not _phase_allowed(config, "mechanistic", set(), authorized=True)
    assert not _phase_allowed(config, "protected", {"A0", "A1", "A2", "A3"}, authorized=True)
    assert not _phase_allowed(
        config,
        "protected",
        {"A0", "A1", "A2", "A3", "execution_authorized", "clean_pushed_commit"},
        authorized=True,
    )
    authorized = copy.deepcopy(config)
    authorized["execution_authorized"] = True
    assert _phase_allowed(
        authorized,
        "protected",
        {"A0", "A1", "A2", "A3", "execution_authorized", "clean_pushed_commit"},
        authorized=True,
    )
    assert set(policy["allowed_statuses"]) == {
        "PREREGISTERED_NOT_AUTHORIZED",
        "IMPLEMENTED",
        "SMOKE_VALIDATED",
        "EXECUTION_BLOCKED",
        "INELIGIBLE_TASK",
        "NEGATIVE_RESULT",
        "POSITIVE_EXPLORATORY",
        "POSITIVE_PROTECTED",
    }
