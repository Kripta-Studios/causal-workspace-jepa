from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from causal_workspace_jepa.data.synthetic.pointmass import step_pointmass
from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_004 import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    EXPERIMENT_ID,
    FORBIDDEN_SEEDS,
    FROZEN_THRESHOLDS,
    INDEPENDENT_CONTROLS,
    LADDER_RUNGS,
    PHYSICS_DEPENDENCY,
    PathAwareActionDeltaPredictor,
    _aggregate,
    _authorize_confirmation,
    _claim_boundary,
    _require_execution_authorized,
    adjudicate_seed,
    classify_path,
    run_development_rung,
)


def test_historical_seeds_are_forbidden() -> None:
    for seed in (43, 47, 53, 59, 71, 73, 79, 83, 89, 1013, 1031, 1033, 1039, 1049, 1051, 1061):
        assert seed in FORBIDDEN_SEEDS


def test_confirmation_seeds_are_not_opened() -> None:
    assert DEVELOPMENT_SEEDS == (97, 101, 107)
    assert CONFIRMATION_SEEDS == (1063, 1069, 1087)
    assert LADDER_RUNGS == (800, 2000, 5000)
    assert not Path("artifacts/metrics/crct_learned_wm_action_delta_v4.json").exists()


def test_config_matches_frozen_thresholds() -> None:
    config = json.loads(
        Path("configs/experiments/crct_learned_wm_action_delta_v4.json").read_text(encoding="utf-8")
    )
    assert config["thresholds"] == FROZEN_THRESHOLDS
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["status"] == "PREREGISTERED_NOT_RUN"
    assert config["execution_authorized"] is True
    assert "action-stem MSRS cannot be Level 3" in config["claim_boundary"]
    assert "REDUNDANT_ROUTES is not Level 3" in config["claim_boundary"]
    assert config["interacting_is_level3_pass"] is False
    assert config["redundant_routes_is_level3_pass"] is False
    assert config["redundant_routes_opens_confirmation"] is False
    assert config["action_stem_msrs_cannot_be_level3"] is True
    assert config["seed_59_retrospective_pass"] is False
    assert config["independent_controls"] == ["dvy", "dy"]


def test_claim_not_jepa_and_interacting_not_level3() -> None:
    text = _claim_boundary()
    assert "not a JEPA" in text
    assert "INTERACTING is not Level 3" in text
    assert "action-stem MSRS cannot be Level 3" in text


def test_interacting_is_not_a_level3_class() -> None:
    assert classify_path({"full": 0.99, "skip": 0.1, "residual": 0.1}) == "INTERACTING"
    assert classify_path({"full": 0.99, "skip": 0.99, "residual": 0.1}) == "DIRECT"
    assert classify_path({"full": 0.99, "skip": 0.1, "residual": 0.99}) == "DISTRIBUTED"
    assert classify_path({"full": 0.99, "skip": 0.99, "residual": 0.99}) == "REDUNDANT_ROUTES"
    assert classify_path({"full": 0.2, "skip": 0.99, "residual": 0.99}) is None


def test_pointmass_dependency_matches_euler() -> None:
    rng = np.random.default_rng(3)
    base_state = rng.uniform(-0.5, 0.5, size=4).astype(np.float64)
    base_action = np.array([0.2, -0.3], dtype=np.float64)
    eps = 1e-3
    base_delta = step_pointmass(base_state, base_action) - base_state
    names = ("dx", "dy", "dvx", "dvy")
    inputs = {"ax": 0, "ay": 1, "vx": 2, "vy": 3}
    observed = {ch: {} for ch in names}
    for inp, index in inputs.items():
        if inp in {"ax", "ay"}:
            perturbed = base_action.copy()
            perturbed[index] += eps
            delta = step_pointmass(base_state, perturbed) - base_state
        else:
            perturbed = base_state.copy()
            perturbed[index] += eps
            delta = step_pointmass(perturbed, base_action) - perturbed
        change = np.abs(delta - base_delta)
        for i, ch in enumerate(names):
            observed[ch][inp] = "D" if change[i] > 1e-6 else "0"
    assert observed == PHYSICS_DEPENDENCY
    assert INDEPENDENT_CONTROLS == ("dvy", "dy")


def test_g_skip_overwrites_residual_and_g_res_holds_skip1_only() -> None:
    torch.manual_seed(5)
    model = PathAwareActionDeltaPredictor(97)
    state = torch.zeros(8, 4)
    action_a = torch.zeros(8, 2)
    action_b = torch.zeros(8, 2)
    action_b[:, 0] = 0.8
    with torch.no_grad():
        y_a, _sites_a, path_a = model.forward_path(state, action_a, None, None)
        _y_b, sites_b, _ = model.forward_path(state, action_b, None, None)
        patch = {name: sites_b[name] for name in sites_b}
        y_full, _, _ = model.forward_path(state, action_a, patch, None)
        y_skip, _, path_skip = model.forward_path(
            state, action_a, patch, {"hid1": path_a["hid1"], "hid2": path_a["hid2"]}
        )
        y_res, _, _ = model.forward_path(state, action_a, patch, {"skip1": path_a["h0"]})
        y_hold_all, _, _ = model.forward_path(
            state,
            action_a,
            patch,
            {
                "hid1": path_a["hid1"],
                "hid2": path_a["hid2"],
                "skip1": path_a["h0"],
                "skip2": path_a["h1"],
            },
        )
    assert torch.allclose(path_skip["hid1"], path_a["hid1"])
    assert torch.allclose(path_skip["hid2"], path_a["hid2"])
    assert torch.mean((y_full - y_a).square()) > 1e-8
    assert torch.mean((y_hold_all - y_a).square()) < 1e-8
    assert not torch.allclose(y_skip, y_res)
    patched_b1 = [name for name in patch if name.startswith("b1_")]
    if patched_b1:
        name = patched_b1[0]
        index = int(name.split("_")[1])
        y_skip2, sites_skip, _ = model.forward_path(
            state, action_a, patch, {"hid1": path_a["hid1"], "hid2": path_a["hid2"]}
        )
        assert torch.allclose(sites_skip[name], path_a["hid1"][:, index])
        assert not torch.allclose(sites_skip[name], patch[name])
        _ = y_skip2


def test_confirmation_rejects_003_and_inconclusive(tmp_path: Path) -> None:
    bad_003 = tmp_path / "p003.json"
    bad_003.write_text(
        json.dumps(
            {
                "experiment_id": "CRCT-LEARNED-WM-ACTION-DELTA-003",
                "stage": "development",
                "status": "PATH_MECHANISM_RECOVERY_PASSED",
                "all_seeds_passed": True,
                "all_seeds_competent": True,
            }
        ),
        encoding="utf-8",
    )
    try:
        _authorize_confirmation(bad_003.as_posix())
    except ValueError:
        pass
    else:
        raise AssertionError("003 artifact authorized 004 confirmation")
    mixed = tmp_path / "mixed.json"
    mixed.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "stage": "development",
                "status": "INCONCLUSIVE",
                "all_seeds_passed": False,
                "all_seeds_competent": True,
                "shared_path_class": None,
                "seeds": [97, 101, 107],
                "threshold_digest": "x",
                "source_digest": "y",
                "selected_rung": 800,
                "train_steps": 800,
            }
        ),
        encoding="utf-8",
    )
    try:
        _authorize_confirmation(mixed.as_posix())
    except ValueError:
        return
    raise AssertionError("inconclusive development authorized confirmation")


def test_execution_authorized_after_freeze() -> None:
    _require_execution_authorized()


def test_climb_requires_previous() -> None:
    try:
        run_development_rung(2000)
    except ValueError as exc:
        assert "previous_path" in str(exc) or "require-previous" in str(exc) or "climbing" in str(exc)
        return
    raise AssertionError("rung 2000 ran without previous_path")


def test_003_and_002_statuses_unchanged() -> None:
    v3 = json.loads(
        Path("artifacts/metrics/crct_learned_wm_action_delta_v3.dev.json").read_text(encoding="utf-8")
    )
    v2 = json.loads(
        Path("artifacts/metrics/crct_learned_wm_action_delta_v2.rung800.json").read_text(
            encoding="utf-8"
        )
    )
    assert v3["status"] == "MODEL_INCOMPETENT"
    assert v2["status"] == "INCONCLUSIVE"
    assert v2["rows"][0]["status"] == "ARCHITECTURE_CUTSET"


def test_interacting_cannot_be_experiment_pass() -> None:
    rows = [
        {
            "seed": seed,
            "status": "MEDIATOR_FOUND_PATH_UNRESOLVED",
            "path_class": "INTERACTING",
            "msrs": ["act_0", "b1_0"],
        }
        for seed in DEVELOPMENT_SEEDS
    ]
    payload = _aggregate(rows, "development", 2000)
    assert payload["status"] == "MEDIATOR_FOUND_PATH_UNRESOLVED"
    assert payload["all_seeds_passed"] is False
    assert payload["evidence_level"] == "None"
    assert payload["shared_path_class"] is None


def test_mixed_direct_distributed_is_inconclusive() -> None:
    rows = [
        {
            "seed": 97,
            "status": "DIRECT_PATH_MECHANISM_PASSED",
            "path_class": "DIRECT",
            "msrs": ["b1_0"],
        },
        {
            "seed": 101,
            "status": "DISTRIBUTED_PATH_MECHANISM_PASSED",
            "path_class": "DISTRIBUTED",
            "msrs": ["b1_0"],
        },
        {
            "seed": 107,
            "status": "DIRECT_PATH_MECHANISM_PASSED",
            "path_class": "DIRECT",
            "msrs": ["b1_1"],
        },
    ]
    payload = _aggregate(rows, "development", 2000)
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["all_seeds_passed"] is False
    assert payload["functional_convergence"] is False


def test_confirmation_path_class_must_match_development() -> None:
    rows = [
        {
            "seed": seed,
            "status": "DIRECT_PATH_MECHANISM_PASSED",
            "path_class": "DIRECT",
            "msrs": [f"b1_{i}"],
        }
        for i, seed in enumerate(CONFIRMATION_SEEDS)
    ]
    payload = _aggregate(
        rows,
        "confirmation",
        2000,
        required_shared_path_class="DISTRIBUTED",
    )
    assert payload["status"] == "CONFIRMATION_PATH_CLASS_MISMATCH"
    assert payload["all_seeds_passed"] is False
    assert payload["h_equivalent"] is False
    matched = _aggregate(
        rows,
        "confirmation",
        2000,
        required_shared_path_class="DIRECT",
    )
    assert matched["status"] == "PATH_MECHANISM_RECOVERY_PASSED"
    assert matched["shared_path_class"] == "DIRECT"
    assert matched["h_equivalent"] is True
    assert matched["functional_convergence"] is True


def _adjudicate_ok(**overrides):
    payload = {
        "coalition": ["act_0", "b1_0"],
        "sufficiency_dvx": 0.01,
        "drop_still_sufficient": False,
        "necessity_dvx": 0.5,
        "spec_failed": False,
        "random_sufficient": 0,
        "act_random_sufficient": 0,
        "g_full": 0.99,
        "gauge_fn": 1e-12,
        "g_suff": 0.01,
        "g_nec": 0.5,
        "g_path": "DIRECT",
        "path_class": "DIRECT",
        "action_only": False,
        "probe_unique_fail": False,
    }
    payload.update(overrides)
    return adjudicate_seed(**payload)


def test_action_stem_direct_is_gateway_not_level3() -> None:
    status, level = _adjudicate_ok(
        coalition=["act_2", "act_0"],
        action_only=True,
        path_class="DIRECT",
        g_path="DIRECT",
    )
    assert status == "INFORMATION_GATEWAY_ONLY"
    assert level == 2


def test_residual_direct_can_be_level3() -> None:
    status, level = _adjudicate_ok()
    assert status == "DIRECT_PATH_MECHANISM_PASSED"
    assert level == 3


def test_redundant_routes_is_level2_and_not_experiment_pass() -> None:
    status, level = _adjudicate_ok(
        path_class="REDUNDANT_ROUTES",
        g_path="REDUNDANT_ROUTES",
    )
    assert status == "REDUNDANT_ROUTES"
    assert level == 2
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
    assert payload["shared_path_class"] is None
    assert payload["functional_convergence"] is False


def test_gauged_necessity_is_a_gate() -> None:
    status, level = _adjudicate_ok(g_nec=0.01)
    assert status == "GAUGE_FAILED"
    assert level == 0


def test_confirmation_rejects_redundant_routes(tmp_path: Path) -> None:
    path = tmp_path / "red.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "stage": "development",
                "status": "REDUNDANT_ROUTES",
                "all_seeds_passed": False,
                "all_seeds_competent": True,
                "shared_path_class": "REDUNDANT_ROUTES",
                "seeds": [97, 101, 107],
            }
        ),
        encoding="utf-8",
    )
    try:
        _authorize_confirmation(path.as_posix())
    except ValueError:
        return
    raise AssertionError("REDUNDANT_ROUTES authorized confirmation")
