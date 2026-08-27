from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_005 import (
    EXPERIMENT_ID as ID005,
)
from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_conditional_mediator_006 import (
    CONFIG_PATH,
    DEVELOPMENT_SEEDS,
    FROZEN_THRESHOLDS,
    EXPERIMENT_ID,
    FORBIDDEN_SEEDS,
    INDEPENDENT_CONTROLS,
    PARENT_ID,
    CONFIRMATION_SEEDS,
    ConditionalPlantedPredictor,
    _authorize_confirmation,
    _require_execution_authorized,
    adjudicate_seed,
    classify_conditional_downstream,
    conditional_downstream_report,
    critical_early_carrier_f1,
    mechanism_hierarchy,
    msrs_early_bottleneck_bias,
    n_down,
    ontology_identifiability,
    planted_suite,
    refined_mechanism_tuple,
    run_learned_development,
    s_down,
)


def test_005_unchanged_and_006_is_inconclusive() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["status"] == "INCONCLUSIVE"
    assert config["execution_authorized"] is False
    assert config["confirmation"] == "CLOSED"
    assert config["selected_rung"] == 800
    assert config["learned_models_trained"] is True
    assert config["parent"] == PARENT_ID
    assert config["parent_status_preserved"] == "INCONCLUSIVE"
    assert config["require_residual_units_by_fiat"] is False
    assert config["level3_authorized"] is False
    assert config["cached_r2_assigns_class"] is False
    assert config["thresholds"] == FROZEN_THRESHOLDS
    assert config["draft_thresholds"] == FROZEN_THRESHOLDS
    assert INDEPENDENT_CONTROLS == ("dvy", "dy")
    v5 = json.loads(Path("configs/experiments/crct_learned_wm_action_delta_v5.json").read_text())
    assert v5["experiment_id"] == ID005
    assert v5["status"] == "INCONCLUSIVE"
    assert v5["execution_authorized"] is False
    assert v5["confirmation"] == "CLOSED"
    rung = json.loads(
        Path("artifacts/metrics/crct_learned_wm_action_delta_v5.rung800.json").read_text(
            encoding="utf-8"
        )
    )
    assert rung["status"] == "INCONCLUSIVE"
    assert rung["evidence_level"] == "None"
    assert all(row.get("stage_b_ran") is False for row in rung["rows"])
    assert not Path("artifacts/metrics/crct_learned_wm_conditional_mediator_v6.json").exists()
    assert not Path("artifacts/metrics/crct_learned_wm_action_delta_v6.json").exists()
    assert not Path(
        "artifacts/metrics/crct_learned_wm_conditional_mediator_v6.rung2000.json"
    ).exists()
    assert not Path(
        "artifacts/metrics/crct_learned_wm_conditional_mediator_v6.rung5000.json"
    ).exists()


def test_proposed_seeds_are_fresh() -> None:
    assert DEVELOPMENT_SEEDS == (173, 179, 181)
    assert CONFIRMATION_SEEDS == (1171, 1181, 1187)
    for seed in (43, 47, 53, 59, 79, 97, 101, 107, 109, 113, 127, 1103, 1109, 1117):
        assert seed in FORBIDDEN_SEEDS
    for seed in DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS:
        assert seed not in FORBIDDEN_SEEDS


def test_learned_execution_is_forbidden() -> None:
    with pytest.raises(ValueError, match="not frozen|not authorized"):
        run_learned_development()
    with pytest.raises(ValueError, match="not frozen|not authorized"):
        _require_execution_authorized()


def test_development_outcome_is_inconclusive() -> None:
    rung800 = json.loads(
        Path("artifacts/metrics/crct_learned_wm_conditional_mediator_v6.rung800.json").read_text(
            encoding="utf-8"
        )
    )
    assert rung800["status"] == "INCONCLUSIVE"
    assert rung800["all_seeds_competent"] is True
    assert rung800["all_seeds_passed"] is False
    assert rung800["evidence_level"] == "None"
    assert rung800["selected_rung"] == 800
    assert rung800["shared_downstream_class"] is None
    assert rung800["rows"][0]["status"] == "SUFFICIENCY_FAILED"
    assert rung800["rows"][1]["status"] == "SPECIFICITY_FAILED"
    assert rung800["rows"][2]["status"] == "SUFFICIENCY_FAILED"
    assert all(row.get("stage_2b_ran") is False for row in rung800["rows"])
    assert all(row.get("downstream_class") is None for row in rung800["rows"])
    assert not Path("artifacts/metrics/crct_learned_wm_conditional_mediator_v6.json").exists()
    v5 = json.loads(
        Path("artifacts/metrics/crct_learned_wm_action_delta_v5.rung800.json").read_text(
            encoding="utf-8"
        )
    )
    assert v5["status"] == "INCONCLUSIVE"


def test_execution_closed_after_inconclusive() -> None:
    try:
        _require_execution_authorized()
    except ValueError:
        return
    raise AssertionError("006 execution still authorized after INCONCLUSIVE")


def test_n_down_and_s_down_equations() -> None:
    assert n_down(0.0, 1.0) == pytest.approx(1.0)
    assert n_down(1.0, 1.0) == pytest.approx(0.0)
    assert math.isnan(n_down(0.2, 0.0))
    assert s_down(1.0, 0.0, 1.0) == pytest.approx(1.0)
    assert math.isnan(s_down(1.0, 1.0, 1.0))


def test_planted_suite_distinguishes_all_six_cases() -> None:
    suite = planted_suite(seed=173)
    expected = {
        "early_carrier_f1": "DOWNSTREAM_F1",
        "early_carrier_f2": "DOWNSTREAM_F2",
        "early_carrier_f1_f2": "DOWNSTREAM_F1_F2",
        "true_direct": "DIRECT",
        "redundant_downstream": "REDUNDANT_DOWNSTREAM",
        "interacting_downstream": "INTERACTING_DOWNSTREAM",
    }
    for mode, path_class in expected.items():
        row = suite[mode]
        assert row["path_class"] == path_class, (mode, row)
        assert row["msrs_recompute_finds_v_up"] is True
        assert row["specificity"]["failed"] is False
        assert row["specificity"]["dx_downstream_not_a_negative_control"] is True


def test_critical_early_carrier_f1_separates_stem_from_f1() -> None:
    row = critical_early_carrier_f1(seed=173)
    assert row["critical_pass"] is True
    assert row["path_class"] == "DOWNSTREAM_F1"
    assert row["old_stage_a_recompute_object"] == "action_stem"
    assert row["new_conditional_object"] == "r1"
    assert row["msrs_recompute_finds_v_up"] is True
    assert row["msrs_meanfill_act_only"] is False
    assert row["n_r1"] >= FROZEN_THRESHOLDS["n_down_min"]
    assert row["n_r2"] < FROZEN_THRESHOLDS["n_down_min"]


def test_redundant_and_interacting_are_not_the_same() -> None:
    suite = planted_suite(seed=173)
    red = suite["redundant_downstream"]
    inter = suite["interacting_downstream"]
    assert red["path_class"] == "REDUNDANT_DOWNSTREAM"
    assert inter["path_class"] == "INTERACTING_DOWNSTREAM"
    assert red["n_r1"] < FROZEN_THRESHOLDS["n_down_min"]
    assert red["n_r2"] < FROZEN_THRESHOLDS["n_down_min"]
    assert red["s_r1_hold"] >= FROZEN_THRESHOLDS["s_down_min"]
    assert red["s_r2_hold"] >= FROZEN_THRESHOLDS["s_down_min"]
    assert inter["n_r1"] >= FROZEN_THRESHOLDS["n_down_min"]
    assert inter["n_r2"] >= FROZEN_THRESHOLDS["n_down_min"]
    assert inter["s_r1_hold"] < FROZEN_THRESHOLDS["s_down_min"]
    assert inter["s_r2_hold"] < FROZEN_THRESHOLDS["s_down_min"]


def test_true_direct_does_not_require_residuals() -> None:
    row = planted_suite(seed=173)["true_direct"]
    assert row["path_class"] == "DIRECT"
    assert row["msrs_meanfill_act_only"] is True
    assert row["g_hold_both"] >= FROZEN_THRESHOLDS["counterfactual_gap_min"]


def test_ontology_b_is_identifiable_and_gauge_role_survives() -> None:
    probe = ontology_identifiability(seed=173)
    assert probe["preferred_ontology"] == "B"
    assert probe["ontology_a_node_v_down_identifiable"] is False
    assert probe["ontology_b_branch_message_v_down_identifiable"] is True
    assert probe["unit_r1_hold_mse"] == pytest.approx(0.0)
    assert probe["f2_skip2_hold_gap"] >= FROZEN_THRESHOLDS["counterfactual_gap_min"]
    assert probe["f2_r2_hold_gap"] < FROZEN_THRESHOLDS["counterfactual_gap_min"]
    assert probe["skip2_hold_vs_r2_hold_mse"] > 1e-3
    assert probe["decoy_b1_5_still_transmits_g_v"] >= FROZEN_THRESHOLDS["counterfactual_gap_min"]
    assert probe["f1_class"] == "DOWNSTREAM_F1"
    assert probe["f1_gauge_class"] == "DOWNSTREAM_F1"
    assert probe["gauge_downstream_role_survives"] is True


def test_msrs_bias_is_recompute_not_meanfill() -> None:
    bias = msrs_early_bottleneck_bias()
    assert bias["recompute_or_g_full_biased_toward_early_bottlenecks"] is True
    assert bias["global_meanfill_msrs_biased_toward_early_bottlenecks"] is False
    assert "intact downstream" in bias["causal_reason"]


def test_hierarchy_and_tuple_refinement() -> None:
    hier = mechanism_hierarchy()
    assert hier["L2B_V_down"].startswith("conditional downstream")
    tuple_ = refined_mechanism_tuple()
    assert tuple_["refined"] == "M = (C, V_up, V_down, E, I)"
    assert "branch-message" in tuple_["V_down"]


def test_classifier_rejects_low_g_v() -> None:
    assert (
        classify_conditional_downstream(
            {
                "g_v": 0.2,
                "g_damaged": 0.0,
                "n_r1": 1.0,
                "n_r2": 0.0,
                "s_r1_hold": 1.0,
                "s_r2_hold": 0.0,
                "s_r1_desc": 1.0,
                "s_both": 1.0,
                "interaction": 0.0,
            }
        )
        is None
    )
    assert (
        classify_conditional_downstream(
            {
                "g_v": float("nan"),
                "g_damaged": 0.0,
                "n_r1": 1.0,
                "n_r2": 0.0,
                "s_r1_hold": 1.0,
                "s_r2_hold": 0.0,
                "s_r1_desc": 1.0,
                "s_both": 1.0,
                "interaction": 0.0,
            }
        )
        is None
    )


def test_unknown_plant_rejected() -> None:
    with pytest.raises(ValueError, match="unknown plant"):
        ConditionalPlantedPredictor("force_residual_units")


def _adjudicate_ok(**overrides):
    payload = {
        "coalition": ["act_0", "act_1"],
        "sufficiency_dvx": 0.01,
        "drop_still_sufficient": False,
        "necessity_dvx": 0.5,
        "spec_failed": False,
        "random_sufficient": 0,
        "act_random_sufficient": 0,
        "g_v": 0.99,
        "gauge_fn": 1e-12,
        "g_suff": 0.01,
        "g_nec": 0.5,
        "g_path": "DOWNSTREAM_F1",
        "path_class": "DOWNSTREAM_F1",
        "branch_control_failed": False,
        "stage_2b_ran": True,
    }
    payload.update(overrides)
    return adjudicate_seed(**payload)


def test_action_stem_can_be_level2b_not_gateway() -> None:
    status, level = _adjudicate_ok()
    assert status == "DOWNSTREAM_F1_MEDIATION_PASSED"
    assert level == 2


def test_residual_membership_is_not_required() -> None:
    status, level = _adjudicate_ok(coalition=["act_0"])
    assert status == "DOWNSTREAM_F1_MEDIATION_PASSED"
    assert level == 2


def test_stage_a_failure_does_not_open_stage_2b() -> None:
    status, level = _adjudicate_ok(
        sufficiency_dvx=0.2,
        stage_2b_ran=False,
        path_class=None,
        g_path=None,
    )
    assert status == "SUFFICIENCY_FAILED"
    assert level == 0


def test_direct_and_unresolved_are_not_level3() -> None:
    status, level = _adjudicate_ok(path_class="DIRECT", g_path="DIRECT")
    assert status == "DIRECT_TRANSMISSION_PASSED"
    assert level == 2
    status, level = _adjudicate_ok(
        path_class="DOWNSTREAM_UNRESOLVED", g_path="DOWNSTREAM_UNRESOLVED"
    )
    assert status == "DOWNSTREAM_UNRESOLVED"
    assert level == 2


def test_confirmation_rejects_unpassed_development(tmp_path: Path) -> None:
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "stage": "development",
                "status": "INCONCLUSIVE",
                "all_seeds_passed": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="confirmation closed"):
        _authorize_confirmation(path.as_posix())


def test_claim_not_jepa_and_not_level3() -> None:
    from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_conditional_mediator_006 import (
        _claim_boundary,
        train_model,
    )

    text = _claim_boundary()
    assert "not a JEPA" in text
    assert "Level 3 is not authorized" in text
    assert "does not require residual-unit membership by fiat" in text
    assert "cached r2_P does not assign" in text
    with pytest.raises(ValueError, match="not frozen|not authorized"):
        train_model(173, torch.zeros(4, 4), torch.zeros(4, 2), torch.zeros(4, 4), 1)


def test_b2_in_v_up_is_preserved_under_r1_hold() -> None:
    plant = ConditionalPlantedPredictor("early_carrier_f2")
    row = conditional_downstream_report(plant, v_up=("b2_0",), seed=173)
    assert row["g_v"] >= FROZEN_THRESHOLDS["counterfactual_gap_min"]
    assert row["n_r1"] < FROZEN_THRESHOLDS["n_down_min"]
    assert row["n_r2"] >= FROZEN_THRESHOLDS["n_down_min"]
    assert row["path_class"] == "DOWNSTREAM_F2"


def test_redundant_gauge_mismatch_is_unstable() -> None:
    status, _level = _adjudicate_ok(
        path_class="REDUNDANT_DOWNSTREAM",
        g_path="INTERACTING_DOWNSTREAM",
    )
    assert status == "DOWNSTREAM_CLASS_GAUGE_UNSTABLE"

