from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.experiments.world_model.leflow_amortize import (
    adjudicate,
    assert_protocol,
    load_json_config,
    reject_forbidden_seed,
    require_mknn_adjudication,
    run_seed,
)
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA
from causal_workspace_jepa.planning.amortized_latent import (
    amortized_latent_plan,
    decode_actions,
    fit_inverse_dynamics,
)
from causal_workspace_jepa.planning.cem import iterative_cem_plan, random_shooting_plan

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/wm_leflow_amortize_v1.json"
PROTOCOL = ROOT / "docs/WM_LEFLOW_AMORTIZE_001_PROTOCOL.md"


def _tiny_config() -> dict:
    config = load_json_config(CONFIG)
    config["environment"]["trajectories"] = 12
    config["environment"]["steps"] = 12
    config["splits"]["train"] = 6
    config["splits"]["development"] = 3
    config["splits"]["confirmation"] = 3
    config["splits"]["n_tasks"] = 3
    config["planning"]["shooting_candidates"] = 8
    config["planning"]["cem_candidates"] = 4
    config["planning"]["cem_iterations"] = 2
    config["planning"]["amortized_rerank_n"] = [1, 4]
    return config


def test_protocol_and_config_are_frozen_before_outcomes() -> None:
    config = load_json_config(CONFIG)
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert_protocol(config)
    assert config["experiment_id"] == "WM-LEFLOW-AMORTIZE-001"
    assert config["not_a_leflow_reproduction"] is True
    assert config["confirmation_seeds"] == [151, 157, 163]
    assert "`151, 157, 163`" in protocol
    assert config["planning"]["primary_horizon"] == 5
    assert config["planning"]["diagnostic_horizons"] == [10]
    assert config["planning"]["amortized_rerank_n"] == [1, 64]
    assert config["gates"]["success_slack"] == 0.05
    assert config["planning"]["success_position_mse_max"] == 0.15
    assert "not OOD" in config["planning"]["horizon_collapse_warning"]
    assert 131 in config["forbidden_seeds"]


def test_forbidden_seeds_and_mknn_gate(tmp_path: Path) -> None:
    config = load_json_config(CONFIG)
    with pytest.raises(ValueError):
        reject_forbidden_seed(131, list(config["forbidden_seeds"]))
    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        json.dumps({"experiment_id": "NOPE", "status": "NEGATIVE_RESULT", "integrity_blockers": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="WM-PLATONIC-MKNN-001"):
        require_mknn_adjudication(wrong)
    payload = {
        "experiment_id": "WM-PLATONIC-MKNN-001",
        "status": "NEGATIVE_RESULT",
        "integrity_blockers": ["p0"],
    }
    path = tmp_path / "mknn.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity_blockers"):
        require_mknn_adjudication(path)


def test_delta_z_control_is_capacity_matched_and_no_delta_decode_uses_zeros() -> None:
    rng = np.random.default_rng(3)
    latents = rng.normal(size=(6, 5, 4)).astype(np.float32)
    actions = rng.normal(size=(6, 4, 2)).astype(np.float32)
    weights_a = fit_inverse_dynamics(latents, actions, include_delta=False, ridge=1e-4)
    weights_b = fit_inverse_dynamics(latents, actions, include_delta=True, ridge=1e-4)
    assert weights_a.shape == weights_b.shape
    dim = 2
    weights = np.zeros((dim + dim + dim + 1, 2), dtype=np.float32)
    weights[4:6] = 100.0
    z_t = np.ones((3, dim), dtype=np.float32)
    z_next = np.ones((3, dim), dtype=np.float32) * 3.0
    assert np.allclose(decode_actions(weights, z_t, z_next, include_delta=False), 0.0)
    assert not np.allclose(decode_actions(weights, z_t, z_next, include_delta=True), 0.0)


def test_n1_n64_and_horizon_labels() -> None:
    dataset = generate_pointmass2d(trajectories=8, steps=12, seed=3)
    model = TinyActionConditionedJEPA.fit(
        dataset.observations[:6],
        dataset.actions[:6],
        latent_dim=8,
        seed=3,
    )
    packed = dataset.observations[:6].reshape(-1, 4)
    encoded = model.encode(packed).tensor.reshape(6, dataset.observations.shape[1], -1)
    weights = fit_inverse_dynamics(encoded, dataset.actions[:6], include_delta=True, ridge=1e-4)
    start = dataset.observations[6, 0]
    goal = dataset.observations[6, 5]
    one = amortized_latent_plan(
        model,
        weights,
        start,
        goal,
        horizon=5,
        candidates=1,
        seed=3,
        noise_std=0.0,
        include_delta=True,
    )
    many = amortized_latent_plan(
        model,
        weights,
        start,
        goal,
        horizon=5,
        candidates=64,
        seed=3,
        noise_std=0.05,
        include_delta=True,
    )
    assert one["candidates_evaluated"] == 1
    assert many["candidates_evaluated"] == 64
    shooting = random_shooting_plan(model, start, goal[:2], horizon=5, candidates=8, seed=3)
    cem = iterative_cem_plan(
        model,
        start,
        goal[:2],
        horizon=10,
        candidates=4,
        iterations=2,
        elite_fraction=0.25,
        seed=3,
    )
    assert shooting["candidates_evaluated"] == 8
    assert cem["iterations"] == 2
    assert cem["actions"].shape[0] == 10


def test_world_model_stays_frozen_and_tasks_are_matched() -> None:
    row = run_seed(_tiny_config(), seed=3)
    assert row["world_model_fingerprint"]
    assert row["horizons"]["5"]["is_primary"] is True
    assert row["horizons"]["10"]["is_ood"] is False
    assert row["horizons"]["5"]["world_model_fingerprint"] == row["world_model_fingerprint"]
    assert row["horizons"]["10"]["world_model_fingerprint"] == row["world_model_fingerprint"]
    for arm in row["horizons"]["5"]["arms"].values():
        assert arm["n_tasks"] == 3.0
        assert arm["horizon"] == 5.0
    assert row["inverse_dynamics_weight_shape"]
    assert row["protected_splits_executed"] == []


def test_adjudicate_uses_frozen_gates_only() -> None:
    config = load_json_config(CONFIG)
    gates_before = json.dumps(config["gates"], sort_keys=True)
    fake_row = {
        "seed": 151,
        "horizons": {
            "5": {
                "arms": {
                    "latent_flow_n64": {"success_rate": 0.1, "mean_wall_clock_s": 1.0},
                    "random_shooting_n64": {"success_rate": 0.9, "mean_wall_clock_s": 0.1},
                }
            }
        },
        "inverse_dynamics_mse_no_delta": 1.0,
        "inverse_dynamics_mse_with_delta": 0.5,
    }
    metrics = adjudicate([fake_row], config)
    assert json.dumps(config["gates"], sort_keys=True) == gates_before
    assert metrics["status"] == "NEGATIVE_RESULT"
    assert metrics["evidence_level"] == "None"
    assert metrics["not_a_leflow_reproduction"] is True
    assert metrics["world_model_frozen"] is True


def test_dirty_git_and_unauthorized_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from causal_workspace_jepa.experiments.world_model import leflow_amortize, platonic_mknn

    monkeypatch.setattr(platonic_mknn, "is_git_dirty", lambda: True)
    with pytest.raises(SystemExit):
        platonic_mknn.main(["--config", str(CONFIG)])
    monkeypatch.setattr(leflow_amortize, "is_git_dirty", lambda: True)
    with pytest.raises(SystemExit):
        leflow_amortize.main(["--config", str(CONFIG)])
