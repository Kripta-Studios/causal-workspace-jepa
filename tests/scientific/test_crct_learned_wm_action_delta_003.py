from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from causal_workspace_jepa.common.provenance import stage_cli_command
from causal_workspace_jepa.data.synthetic.pointmass import step_pointmass
from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_003 import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    EXPERIMENT_ID,
    FORBIDDEN_SEEDS,
    FROZEN_THRESHOLDS,
    INDEPENDENT_CONTROLS,
    MODULE,
    PHYSICS_DEPENDENCY,
    TRAIN_STEPS,
    PathAwareActionDeltaPredictor,
    _authorize_confirmation,
    _claim_boundary,
    _hypothesis,
    classify_path,
    source_digest,
)


def test_002_and_001_seeds_are_forbidden() -> None:
    for seed in (43, 47, 53, 59, 71, 73, 1013, 1031, 1033, 1039):
        assert seed in FORBIDDEN_SEEDS


def test_confirmation_seeds_are_not_opened_by_unit_tests() -> None:
    assert DEVELOPMENT_SEEDS == (79, 83, 89)
    assert CONFIRMATION_SEEDS == (1049, 1051, 1061)
    assert TRAIN_STEPS == 800
    assert not Path("artifacts/metrics/crct_learned_wm_action_delta_v3.json").exists()


def test_config_matches_frozen_thresholds() -> None:
    config = json.loads(
        Path("configs/experiments/crct_learned_wm_action_delta_v3.json").read_text(encoding="utf-8")
    )
    assert config["thresholds"] == FROZEN_THRESHOLDS
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["parent_status_preserved"] == "INCONCLUSIVE"
    assert config["status"] == "MODEL_INCOMPETENT"
    assert config["execution_authorized"] is False
    assert config["confirmation"] == "CLOSED"
    assert config["seed_59_retrospective_pass"] is False
    assert config["architecture_cutset_automatic_fail"] is False
    assert config["independent_controls"] == ["dvy", "dy"]


def test_claim_does_not_call_the_model_jepa() -> None:
    text = _claim_boundary()
    assert "not a JEPA" in text
    assert "seed 59 is not a retrospective pass" in text


def test_action_only_is_not_an_automatic_fail() -> None:
    assert classify_path({"full": 0.99, "skip": 0.99, "residual": 0.1}) == "DIRECT"
    assert _hypothesis("PATH_MECHANISM_RECOVERY_PASSED", "DIRECT") == "H_DIRECT"
    assert INDEPENDENT_CONTROLS == ("dvy", "dy")


def test_pointmass_dependency_matches_euler() -> None:
    rng = np.random.default_rng(3)
    base_state = rng.uniform(-0.5, 0.5, size=4).astype(np.float64)
    base_action = np.array([0.2, -0.3], dtype=np.float64)
    eps = 1e-3
    base_next = step_pointmass(base_state, base_action)
    base_delta = base_next - base_state
    names = ("dx", "dy", "dvx", "dvy")
    inputs = {"ax": 0, "ay": 1, "vx": 2, "vy": 3}

    def _delta_after(action=None, state=None) -> np.ndarray:
        st = base_state if state is None else state
        ac = base_action if action is None else action
        nxt = step_pointmass(st, ac)
        return nxt - st

    observed = {ch: {} for ch in names}
    for inp, index in inputs.items():
        if inp in {"ax", "ay"}:
            perturbed = base_action.copy()
            perturbed[index] += eps
            delta = _delta_after(action=perturbed)
        else:
            perturbed = base_state.copy()
            perturbed[index] += eps
            delta = _delta_after(state=perturbed)
        change = np.abs(delta - base_delta)
        for i, ch in enumerate(names):
            observed[ch][inp] = "D" if change[i] > 1e-6 else "0"
    assert observed == PHYSICS_DEPENDENCY
    assert PHYSICS_DEPENDENCY["dy"]["ax"] == "0"
    assert PHYSICS_DEPENDENCY["dx"]["ax"] == "D"


def test_path_holds_isolate_skip_versus_residual() -> None:
    torch.manual_seed(5)
    model = PathAwareActionDeltaPredictor(79)
    state = torch.zeros(8, 4)
    action_a = torch.zeros(8, 2)
    action_b = torch.zeros(8, 2)
    action_b[:, 0] = 0.8
    with torch.no_grad():
        y_a, _sites_a, path_a = model.forward_path(state, action_a, None, None)
        _y_b, sites_b, _ = model.forward_path(state, action_b, None, None)
        patch = {name: sites_b[name] for name in sites_b}
        y_full, _, _ = model.forward_path(state, action_a, patch, None)
        y_skip, _, _ = model.forward_path(
            state, action_a, patch, {"hid1": path_a["hid1"], "hid2": path_a["hid2"]}
        )
        y_res, _, _ = model.forward_path(
            state, action_a, patch, {"skip1": path_a["h0"]}
        )
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
    assert torch.mean((y_full - y_a).square()) > 1e-8
    assert torch.mean((y_hold_all - y_a).square()) < 1e-8
    assert not torch.allclose(y_skip, y_res)


def test_confirmation_rejects_002_artifact(tmp_path: Path) -> None:
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "CRCT-LEARNED-WM-ACTION-DELTA-002",
                "stage": "development",
                "status": "PATH_MECHANISM_RECOVERY_PASSED",
                "all_seeds_competent": True,
                "all_seeds_passed": True,
            }
        ),
        encoding="utf-8",
    )
    try:
        _authorize_confirmation(path.as_posix())
    except ValueError:
        return
    raise AssertionError("002 development artifact authorized 003 confirmation")


def test_confirmation_cli_does_not_fuse_development() -> None:
    command = stage_cli_command(
        MODULE,
        "confirmation",
        "artifacts/metrics/crct_learned_wm_action_delta_v3.json",
        require_development="artifacts/metrics/crct_learned_wm_action_delta_v3.dev.json",
    )
    assert "--stage confirmation" in command
    assert "--stage development" not in command
    assert "&&" not in command


def test_source_digest_includes_parent_modules() -> None:
    import hashlib

    from causal_workspace_jepa.experiments.cross_domain import crct_jepa_action_delta as p001
    from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta as p002
    from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta_003 as p003

    expected = hashlib.sha256(
        Path(p001.__file__).read_bytes()
        + b"\n"
        + Path(p002.__file__).read_bytes()
        + b"\n"
        + Path(p003.__file__).read_bytes()
    ).hexdigest()
    assert source_digest() == expected


def test_confirmation_rejects_inconclusive_003(tmp_path: Path) -> None:
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "stage": "development",
                "status": "INCONCLUSIVE",
                "all_seeds_competent": True,
                "all_seeds_passed": False,
                "seeds": [79, 83, 89],
                "threshold_digest": "x",
                "source_digest": "y",
                "train_steps": 800,
            }
        ),
        encoding="utf-8",
    )
    try:
        _authorize_confirmation(path.as_posix())
    except ValueError:
        return
    raise AssertionError("inconclusive development authorized confirmation")


def test_development_stopped_at_competence() -> None:
    payload = json.loads(
        Path("artifacts/metrics/crct_learned_wm_action_delta_v3.dev.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["status"] == "MODEL_INCOMPETENT"
    assert payload["evidence_level"] == "None"
    assert payload["all_seeds_competent"] is False
    assert payload["all_seeds_passed"] is False
    assert payload["seed_59_retrospective_pass"] is False
    by_seed = {row["seed"]: row for row in payload["rows"]}
    assert by_seed[79]["status"] == "MODEL_INCOMPETENT"
    assert by_seed[79]["circuit_search_ran"] is False
    assert by_seed[79]["competence"]["nmse"]["dy"] > 0.05
    assert by_seed[83]["status"] == "COMPETENT_NOT_INTERPRETED"
    assert by_seed[89]["status"] == "COMPETENT_NOT_INTERPRETED"
    assert by_seed[83]["circuit_search_ran"] is False
    assert by_seed[89]["circuit_search_ran"] is False


def test_002_status_is_unchanged() -> None:
    payload = json.loads(
        Path("artifacts/metrics/crct_learned_wm_action_delta_v2.rung800.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["rows"][0]["status"] == "ARCHITECTURE_CUTSET"
