from __future__ import annotations

import json
from pathlib import Path

from causal_workspace_jepa.common.provenance import stage_cli_command
from causal_workspace_jepa.experiments.cross_domain.crct_jepa_action_delta import (
    FROZEN_THRESHOLDS as PARENT_THRESHOLDS,
)
from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    EXPERIMENT_ID,
    FORBIDDEN_SEEDS,
    FROZEN_THRESHOLDS,
    LADDER_RUNGS,
    MECHANISTIC_THRESHOLDS,
    MODULE,
    _authorize_previous,
    _claim_boundary,
)


def test_001_seeds_are_forbidden() -> None:
    for seed in (43, 47, 53, 1013, 1019, 1021):
        assert seed in FORBIDDEN_SEEDS


def test_confirmation_seeds_are_not_opened_by_unit_tests() -> None:
    assert DEVELOPMENT_SEEDS == (59, 71, 73)
    assert CONFIRMATION_SEEDS == (1031, 1033, 1039)
    assert LADDER_RUNGS == (200, 800, 2000)


def test_mechanistic_thresholds_match_001() -> None:
    assert MECHANISTIC_THRESHOLDS == dict(PARENT_THRESHOLDS)


def test_config_matches_frozen_thresholds() -> None:
    config = json.loads(
        Path("configs/experiments/crct_learned_wm_action_delta_v2.json").read_text(encoding="utf-8")
    )
    assert config["thresholds"] == FROZEN_THRESHOLDS
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["status"] == "INCONCLUSIVE"
    assert config["execution_authorized"] is False
    assert config["confirmation"] == "CLOSED"
    assert config["substrate"] == "supervised_residual_mlp_not_jepa_objective"


def test_confirmation_artifact_was_not_written() -> None:
    assert not Path("artifacts/metrics/crct_learned_wm_action_delta_v2.json").exists()


def test_rung800_development_is_inconclusive_not_a_pass() -> None:
    payload = json.loads(
        Path("artifacts/metrics/crct_learned_wm_action_delta_v2.rung800.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["evidence_level"] == "None"
    assert payload["all_seeds_competent"] is True
    assert payload["all_seeds_passed"] is False
    assert payload["selected_rung"] == 800
    assert payload["stage"] == "development"
    assert "JEPA objective" in payload["claim_boundary"]
    statuses = {row["seed"]: row["status"] for row in payload["rows"]}
    assert statuses[59] == "ARCHITECTURE_CUTSET"
    assert statuses[71] == "SUFFICIENCY_FAILED"
    assert statuses[73] == "SPECIFICITY_FAILED"
    assert all(row["circuit_search_ran"] is True for row in payload["rows"])
    assert all(row["competence"]["passed"] is True for row in payload["rows"])


def test_rung200_did_not_open_circuit_search() -> None:
    payload = json.loads(
        Path("artifacts/metrics/crct_learned_wm_action_delta_v2.rung200.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "MODEL_INCOMPETENT"
    assert payload["all_seeds_competent"] is False
    assert all(row["status"] == "MODEL_INCOMPETENT" for row in payload["rows"])
    assert all(row.get("circuit_search_ran") is not True for row in payload["rows"])


def test_claim_does_not_call_the_model_jepa() -> None:
    text = _claim_boundary()
    assert "not a JEPA" in text
    assert "JEPA objective" in text


def test_development_cli_is_single_rung() -> None:
    command = stage_cli_command(
        MODULE,
        "development",
        "artifacts/metrics/crct_learned_wm_action_delta_v2.rung200.json",
        extra_args="--rung 200",
    )
    assert "--stage development" in command
    assert "--rung 200" in command
    assert "--stage confirmation" not in command
    assert "&&" not in command


def test_confirmation_sidecar_seed_is_1031() -> None:
    command = stage_cli_command(
        MODULE,
        "confirmation",
        "artifacts/metrics/crct_learned_wm_action_delta_v2.json",
        require_development="artifacts/metrics/crct_learned_wm_action_delta_v2.rung800.json",
    )
    assert "--stage confirmation" in command
    assert "--stage development" not in command
    assert "1031" not in command
    assert "--require-development" in command


def test_authorize_previous_rejects_passed_rung(tmp_path: Path) -> None:
    path = tmp_path / "rung200.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "stage": "development",
                "status": "MECHANISM_RECOVERY_PASSED",
                "train_steps": 200,
                "threshold_digest": "x",
                "source_digest": "y",
            }
        ),
        encoding="utf-8",
    )
    try:
        _authorize_previous(path.as_posix(), 800)
    except ValueError:
        return
    raise AssertionError("passed previous rung was allowed to climb")


def test_callable_climb_requires_previous_path() -> None:
    from causal_workspace_jepa.experiments.cross_domain import crct_jepa_action_delta as parent
    from causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta import (
        run_confirmation,
        run_development_rung,
        source_digest,
    )

    try:
        run_development_rung(800)
    except ValueError:
        pass
    else:
        raise AssertionError("rung 800 ran without previous_path")
    try:
        run_confirmation()  # type: ignore[misc]
    except TypeError:
        pass
    else:
        raise AssertionError("confirmation ran without development path")
    import hashlib

    from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta as wm

    expected = hashlib.sha256(
        Path(parent.__file__).read_bytes() + b"\n" + Path(wm.__file__).read_bytes()
    ).hexdigest()
    assert source_digest() == expected


def test_action_only_coalition_is_architecture_cutset() -> None:
    from causal_workspace_jepa.experiments.cross_domain import crct_jepa_action_delta as parent

    coalition = ["act_0", "act_1"]
    assert set(coalition) <= set(parent.ACT_SITES)
    assert not set(coalition) <= set(parent.B1_SITES)
