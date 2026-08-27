from __future__ import annotations

import json
from pathlib import Path

import torch

from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_004 import (
    EXPERIMENT_ID as ID004,
)
from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_005 import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    EXPERIMENT_ID,
    FORBIDDEN_SEEDS,
    FROZEN_THRESHOLDS,
    INDEPENDENT_CONTROLS,
    LADDER_RUNGS,
    PathAwareActionDeltaPredictor,
    PlantedRoutePredictor,
    _aggregate,
    _authorize_confirmation,
    _claim_boundary,
    _require_execution_authorized,
    adjudicate_seed,
    classify_edge_path,
    compose_output,
    edge_factorial,
    run_development_rung,
)


def _gaps(**overrides: float) -> dict[str, float]:
    payload = {
        "g_v": 0.99,
        "g_skip1": 0.0,
        "g_res1": 0.0,
        "g_skip2": 0.0,
        "g_res2": 0.0,
        "g_f1f2": 0.0,
    }
    payload.update(overrides)
    return payload


def test_historical_and_004_seeds_are_forbidden() -> None:
    for seed in (43, 47, 53, 59, 71, 73, 79, 83, 89, 97, 101, 107, 1063, 1069, 1087):
        assert seed in FORBIDDEN_SEEDS
    assert DEVELOPMENT_SEEDS == (109, 113, 127)
    assert CONFIRMATION_SEEDS == (1103, 1109, 1117)
    assert LADDER_RUNGS == (800, 2000, 5000)
    assert not Path("artifacts/metrics/crct_learned_wm_action_delta_v5.json").exists()


def test_config_is_draft_and_matches_thresholds() -> None:
    config = json.loads(
        Path("configs/experiments/crct_learned_wm_action_delta_v5.json").read_text(encoding="utf-8")
    )
    assert config["thresholds"] == FROZEN_THRESHOLDS
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["confirmation"] == "CLOSED"
    assert config["004_g_skip_assigns_path_class"] is False
    assert config["residual_messages"] == "cached_from_reference_forwards_A_and_P"
    assert config["seed_97_level3_pass"] is False
    assert config["action_stem_msrs_cannot_be_level3"] is True
    assert INDEPENDENT_CONTROLS == ("dvy", "dy")
    assert config["status"] in {"DRAFT_NOT_PREREGISTERED", "PREREGISTERED_NOT_RUN", "INCONCLUSIVE", "MODEL_INCOMPETENT", "PATH_MECHANISM_RECOVERY_PASSED", "REDUNDANT_ROUTES", "MEDIATOR_FOUND_PATH_UNRESOLVED"}
    if config["status"] == "DRAFT_NOT_PREREGISTERED":
        assert config["execution_authorized"] is False


def test_004_status_unchanged() -> None:
    v4 = json.loads(Path("configs/experiments/crct_learned_wm_action_delta_v4.json").read_text())
    assert v4["experiment_id"] == ID004
    assert v4["status"] == "INCONCLUSIVE"
    rung = json.loads(
        Path("artifacts/metrics/crct_learned_wm_action_delta_v4.rung2000.json").read_text(
            encoding="utf-8"
        )
    )
    assert rung["status"] == "INCONCLUSIVE"
    assert rung["rows"][0]["status"] == "INFORMATION_GATEWAY_ONLY"


def test_claim_not_jepa_and_not_004_gskip() -> None:
    text = _claim_boundary()
    assert "not a JEPA" in text
    assert "action-stem MSRS cannot be Level 3" in text
    assert "cached r2_P is not a Level-3 F2 edge" in text
    assert "004 G_skip does not assign 005 path class" in text
    assert "seed 97 is not a retrospective Level-3 pass" in text


def test_plant_classifier_direct() -> None:
    assert classify_edge_path(_gaps(g_skip1=0.99, g_res1=0.1, g_res2=0.1)) == "DIRECT"


def test_plant_classifier_f1() -> None:
    assert classify_edge_path(_gaps(g_res1=0.99, g_skip1=0.1, g_res2=0.1)) == "DISTRIBUTED_F1"


def test_plant_classifier_f2_cached_is_not_level3_class() -> None:
    assert classify_edge_path(_gaps(g_res2=0.99, g_skip1=0.1, g_res1=0.1)) == "F2_CACHED_UNIDENTIFIED"


def test_plant_classifier_combined_only_is_interacting_not_f1f2() -> None:
    assert classify_edge_path(_gaps(g_f1f2=0.99, g_skip1=0.1, g_res1=0.1, g_res2=0.1)) == "INTERACTING"


def test_plant_classifier_redundant_independent_sufficiency() -> None:
    assert classify_edge_path(_gaps(g_skip1=0.99, g_res1=0.99, g_res2=0.2)) == "REDUNDANT_ROUTES"


def test_plant_classifier_interacting() -> None:
    assert classify_edge_path(_gaps(g_v=0.99, g_skip1=0.2, g_res1=0.2, g_res2=0.2)) == "INTERACTING"


def test_low_gv_is_not_a_path_class() -> None:
    assert classify_edge_path(_gaps(g_v=0.2, g_skip1=0.99, g_res1=0.99)) is None


def test_compose_reconstructs_factual_and_does_not_use_hid_holds() -> None:
    torch.manual_seed(5)
    model = PathAwareActionDeltaPredictor(109)
    state = torch.zeros(8, 4)
    action = torch.zeros(8, 2)
    action[:, 0] = 0.5
    with torch.no_grad():
        y, _sites, paths = model.forward_path(state, action, None, None)
        y_rec = compose_output(model, paths["h0"], paths["r1"], paths["r2"])
        y_held, _, path_held = model.forward_path(
            state, action, None, {"hid1": paths["hid1"] * 0, "hid2": paths["hid2"] * 0}
        )
    assert torch.allclose(y_rec, y, atol=1e-5)
    assert not torch.allclose(y_held, y)
    assert torch.allclose(path_held["hid1"], torch.zeros_like(path_held["hid1"]))
    with torch.no_grad():
        r1_from_zero = model.b1_w2(torch.zeros_like(paths["hid1"]))
    assert torch.allclose(path_held["r1"], r1_from_zero, atol=1e-5)


def test_execution_refuses_draft_or_closed() -> None:
    config = json.loads(
        Path("configs/experiments/crct_learned_wm_action_delta_v5.json").read_text(encoding="utf-8")
    )
    if config.get("status") == "DRAFT_NOT_PREREGISTERED" or config.get("execution_authorized") is not True:
        try:
            _require_execution_authorized()
        except ValueError:
            return
        raise AssertionError("005 execution authorized while draft/closed")


def test_climb_2000_requires_previous() -> None:
    from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_005 import (
        CONFIG_PATH as V5,
    )

    config = json.loads(V5.read_text(encoding="utf-8"))
    if config.get("execution_authorized") is not True:
        try:
            run_development_rung(2000)
        except ValueError as exc:
            assert "not authorized" in str(exc) or "not frozen" in str(exc) or "previous_path" in str(exc)
            return
        raise AssertionError("005 rung 2000 ran without authorization")
    try:
        run_development_rung(2000)
    except ValueError as exc:
        assert "previous_path" in str(exc) or "climbing" in str(exc)
        return
    raise AssertionError("rung 2000 ran without previous_path")


def test_confirmation_rejects_004_artifact(tmp_path: Path) -> None:
    path = tmp_path / "p004.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "CRCT-LEARNED-WM-ACTION-DELTA-004",
                "stage": "development",
                "status": "PATH_MECHANISM_RECOVERY_PASSED",
                "all_seeds_passed": True,
                "shared_path_class": "DIRECT",
            }
        ),
        encoding="utf-8",
    )
    try:
        _authorize_confirmation(path.as_posix())
    except ValueError:
        return
    raise AssertionError("004 artifact authorized 005 confirmation")


def test_redundant_is_not_experiment_pass() -> None:
    rows = [
        {
            "seed": seed,
            "status": "REDUNDANT_ROUTES",
            "path_class": "REDUNDANT_ROUTES",
            "msrs": ["act_0", "b1_0"],
        }
        for seed in DEVELOPMENT_SEEDS
    ]
    payload = _aggregate(rows, "development", 2000)
    assert payload["status"] == "REDUNDANT_ROUTES"
    assert payload["all_seeds_passed"] is False
    assert payload["evidence_level"] == "None"
    assert payload["functional_convergence"] is False


def test_mixed_f1_f2_is_inconclusive() -> None:
    rows = [
        {
            "seed": 109,
            "status": "DISTRIBUTED_F1_PATH_MECHANISM_PASSED",
            "path_class": "DISTRIBUTED_F1",
            "msrs": ["b1_0"],
        },
        {
            "seed": 113,
            "status": "MEDIATOR_FOUND_PATH_UNRESOLVED",
            "path_class": "F2_CACHED_UNIDENTIFIED",
            "msrs": ["b2_0"],
        },
        {
            "seed": 127,
            "status": "DISTRIBUTED_F1_PATH_MECHANISM_PASSED",
            "path_class": "DISTRIBUTED_F1",
            "msrs": ["b1_1"],
        },
    ]
    payload = _aggregate(rows, "development", 2000)
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["all_seeds_passed"] is False
    assert payload["shared_path_class"] is None


def test_shared_residual_direct_can_pass() -> None:
    rows = [
        {
            "seed": seed,
            "status": "DIRECT_PATH_MECHANISM_PASSED",
            "path_class": "DIRECT",
            "msrs": [f"b1_{i}"],
        }
        for i, seed in enumerate(DEVELOPMENT_SEEDS)
    ]
    payload = _aggregate(rows, "development", 2000)
    assert payload["status"] == "PATH_MECHANISM_RECOVERY_PASSED"
    assert payload["shared_path_class"] == "DIRECT"
    assert payload["evidence_level"] == "Causal effect"
    assert payload["functional_convergence"] is True


def _adjudicate_ok(**overrides):
    payload = {
        "coalition": ["act_0", "b1_0"],
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
        "g_path": "DIRECT",
        "path_class": "DIRECT",
        "action_only": False,
        "edge_control_failed": False,
        "stage_b_ran": True,
    }
    payload.update(overrides)
    return adjudicate_seed(**payload)


def test_interacting_is_not_level3() -> None:
    status, level = _adjudicate_ok(path_class="INTERACTING", g_path="INTERACTING")
    assert status == "MEDIATOR_FOUND_PATH_UNRESOLVED"
    assert level == 2


def test_residual_inclusive_direct_can_be_level3() -> None:
    status, level = _adjudicate_ok()
    assert status == "DIRECT_PATH_MECHANISM_PASSED"
    assert level == 3


def test_edge_factorial_plants_known_routes() -> None:
    cases = (
        ("direct", ["act_0"], "DIRECT"),
        ("f1", ["b1_0"], "DISTRIBUTED_F1"),
        ("f2", ["act_0"], "F2_CACHED_UNIDENTIFIED"),
        ("redundant_skip_f1", ["act_0"], "REDUNDANT_ROUTES"),
        ("interacting", ["act_0"], "INTERACTING"),
        ("f1_copied_by_f2", ["b1_1"], "F2_CACHED_UNIDENTIFIED"),
    )
    for mode, coalition, expected in cases:
        plant = PlantedRoutePredictor(mode)
        gaps = edge_factorial(plant, coalition, seed=109)
        assert gaps["g_skip2_is_alias_of_both1"] < 1e-10, mode
        assert classify_edge_path(gaps) == expected, (mode, gaps)


def test_f1_copied_by_f2_cannot_be_level3() -> None:
    status, level = _adjudicate_ok(
        coalition=["b1_1"],
        action_only=False,
        path_class="F2_CACHED_UNIDENTIFIED",
        g_path="F2_CACHED_UNIDENTIFIED",
    )
    assert status == "MEDIATOR_FOUND_PATH_UNRESOLVED"
    assert level == 2


def test_action_stem_direct_is_gateway_not_level3() -> None:
    status, level = _adjudicate_ok(
        coalition=["act_0", "act_1"],
        action_only=True,
        path_class="DIRECT",
        g_path="DIRECT",
    )
    assert status == "INFORMATION_GATEWAY_ONLY"
    assert level == 2


def test_stage_a_without_stage_b_is_gateway_if_action_stem() -> None:
    status, level = _adjudicate_ok(
        coalition=["act_0", "act_1"],
        action_only=True,
        stage_b_ran=False,
        path_class=None,
        g_path=None,
    )
    assert status == "INFORMATION_GATEWAY_ONLY"
    assert level == 2


def test_edge_control_blocks_level3() -> None:
    status, level = _adjudicate_ok(edge_control_failed=True)
    assert status == "EDGE_CONTROL_FAILED"
    assert level == 0
