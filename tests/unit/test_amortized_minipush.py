from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal_workspace_jepa.data.splits import deterministic_named_split_ids
from causal_workspace_jepa.data.synthetic.minipush import (
    constructed_goal_observation,
    generate_minipush,
    minipush_rollout_state,
    quantize_minipush_actions,
    step_minipush,
)
from causal_workspace_jepa.experiments.world_model.amortized_minipush import (
    FORBIDDEN_SPLIT_NAMES,
    REQUIRED_ARM_FIELDS,
    _require_arm_schema,
    _tasks,
    adjudicate,
    assert_protocol,
    load_json_config,
    qualify_development,
    reject_forbidden_seed,
    require_qualification_passed,
    run_seed,
)
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA
from causal_workspace_jepa.planning.amortized_latent import amortized_latent_plan, fit_inverse_dynamics
from causal_workspace_jepa.planning.cem import iterative_cem_plan, random_shooting_plan
from causal_workspace_jepa.planning.costs import squared_latent_goal_cost

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/wm_amortized_minipush_v1.json"
PROTOCOL = ROOT / "docs/WM_AMORTIZED_PLANNING_MINIPUSH_002_PROTOCOL.md"
T2_METRICS = ROOT / "artifacts/metrics/wm_leflow_amortize_v1.json"
MODULE = ROOT / "src/causal_workspace_jepa/experiments/world_model/amortized_minipush.py"


def _tiny_config() -> dict:
    config = load_json_config(CONFIG)
    config["environment"]["trajectories"] = 12
    config["environment"]["steps"] = 12
    config["splits"]["train"] = 6
    config["splits"]["development"] = 3
    config["splits"]["confirmation"] = 3
    config["splits"]["n_development_tasks"] = 3
    config["splits"]["n_confirmation_tasks"] = 3
    config["planning"]["shooting_candidates"] = 8
    config["planning"]["cem_candidates"] = 4
    config["planning"]["cem_iterations"] = 2
    config["planning"]["amortized_rerank_n"] = [1, 4]
    return config


def test_protocol_and_config_are_frozen_before_outcomes() -> None:
    config = load_json_config(CONFIG)
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert_protocol(config)
    assert config["experiment_id"] == "WM-AMORTIZED-PLANNING-MINIPUSH-002"
    assert config["confirmation_seeds"] == [251, 257, 263]
    assert config["qualification_seed"] == 241
    assert "`251, 257, 263`" in protocol
    assert config["planning"]["success_object_l2_max"] == 1.5
    assert config["planning"]["cost_mode"] == "latent_goal"
    assert config["gates"]["primary_arm"] == "latent_flow_n1"
    assert config["gates"]["search_baseline"] == "iterative_cem"
    assert config["gates"]["success_slack_vs_cem"] == 0.10
    assert 151 in config["forbidden_seeds"]
    assert 131 in config["forbidden_seeds"]
    assert config["not_levljepa_factorial"] is True
    assert "WM-PLATONIC-STITCH" not in MODULE.read_text(encoding="utf-8")


def test_splits_are_disjoint_and_protected_names_fail() -> None:
    ids = deterministic_named_split_ids(96, 277, train=56, development=16, confirmation=24)
    assert set(ids) == {"train", "development", "confirmation"}
    assert not any(name in FORBIDDEN_SPLIT_NAMES for name in ids)
    train, dev, conf = set(ids["train"].tolist()), set(ids["development"].tolist()), set(ids["confirmation"].tolist())
    assert train.isdisjoint(dev) and train.isdisjoint(conf) and dev.isdisjoint(conf)
    with pytest.raises(ValueError):
        reject_forbidden_seed(151, load_json_config(CONFIG)["forbidden_seeds"])


def test_minipush_rollout_matches_generator_and_quantize_is_cardinal() -> None:
    dataset = generate_minipush(trajectories=3, steps=8, seed=5)
    replayed = minipush_rollout_state(dataset.states[0, 0], dataset.actions[0])
    np.testing.assert_allclose(replayed, dataset.states[0, -1])
    quantized = quantize_minipush_actions(np.array([[0.9, 0.1], [-0.2, -0.8]], dtype=np.float32))
    np.testing.assert_array_equal(quantized, np.array([[1, 0], [0, -1]], dtype=np.float32))
    stepped = step_minipush(np.array([5, 5, 6, 5, 10, 10], dtype=np.float32), np.array([1.0, 0.0]))
    assert stepped.shape == (6,)


def test_identical_goal_information_for_every_query() -> None:
    dataset = generate_minipush(trajectories=4, steps=6, seed=5)
    tasks = _tasks(dataset.states)
    for task in tasks:
        goal = task["goal_observation"]
        start = task["start"]
        np.testing.assert_allclose(goal, constructed_goal_observation(start))
        np.testing.assert_allclose(goal[:2], start[:2])
        np.testing.assert_allclose(goal[2:4], start[4:6])
        np.testing.assert_allclose(goal[4:6], start[4:6])
        assert goal.shape == (6,)
    source = MODULE.read_text(encoding="utf-8")
    assert 'task["goal_observation"]' in source
    assert "goal_position" not in source
    assert 'cost_mode="latent_goal"' in source


def test_latent_goal_cost_is_shared_and_not_xy_slice() -> None:
    predicted = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    z_goal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(squared_latent_goal_cost(predicted, z_goal), 0.0)
    dataset = generate_minipush(trajectories=6, steps=8, seed=5)
    model = TinyActionConditionedJEPA.fit(
        dataset.states[:4].astype(np.float32),
        dataset.actions[:4].astype(np.float32),
        latent_dim=8,
        seed=5,
    )
    start = dataset.states[4, 0]
    goal = constructed_goal_observation(start)
    shooting = random_shooting_plan(
        model,
        start,
        goal,
        horizon=5,
        candidates=8,
        seed=5,
        cost_mode="latent_goal",
        quantize_fn=quantize_minipush_actions,
    )
    cem = iterative_cem_plan(
        model,
        start,
        goal,
        horizon=5,
        candidates=4,
        iterations=2,
        elite_fraction=0.25,
        seed=5,
        cost_mode="latent_goal",
        quantize_fn=quantize_minipush_actions,
    )
    assert shooting["candidates_evaluated"] == 8
    assert cem["candidates_evaluated"] == 8
    assert cem["iterations"] == 2
    packed = dataset.states[:4].reshape(-1, 6)
    encoded = model.encode(packed).tensor.reshape(4, 8, -1)
    weights = fit_inverse_dynamics(encoded, dataset.actions[:4].astype(np.float32), include_delta=True, ridge=1e-4)
    one = amortized_latent_plan(
        model,
        weights,
        start,
        goal,
        horizon=5,
        candidates=1,
        seed=5,
        noise_std=0.0,
        cost_mode="latent_goal",
        quantize_fn=quantize_minipush_actions,
    )
    many = amortized_latent_plan(
        model,
        weights,
        start,
        goal,
        horizon=10,
        candidates=4,
        seed=5,
        noise_std=0.05,
        cost_mode="latent_goal",
        quantize_fn=quantize_minipush_actions,
    )
    assert one["candidates_evaluated"] == 1
    assert many["candidates_evaluated"] == 4
    assert many["actions"].shape[0] == 10


def test_world_model_stays_frozen_and_metrics_are_persisted() -> None:
    row = run_seed(_tiny_config(), seed=3, eval_split="development")
    assert row["eval_split"] == "development"
    assert row["confirmation_metrics_computed"] is False
    assert row["horizons"]["5"]["world_model_fingerprint"] == row["world_model_fingerprint"]
    assert row["horizons"]["10"]["world_model_fingerprint"] == row["world_model_fingerprint"]
    assert row["horizons"]["5"]["cost_mode"] == "latent_goal"
    arms = row["horizons"]["5"]["arms"]
    assert set(arms) == {
        "random_shooting_n64",
        "iterative_cem",
        "latent_flow_n1",
        "latent_flow_n64",
        "action_flow_n1",
    }
    for arm in arms.values():
        _require_arm_schema(arm)
        assert len(arm["goal_distances"]) == 3
        assert arm["failure_rate"] == pytest.approx(1.0 - arm["success_rate"])
    assert arms["iterative_cem"]["mean_wm_rollout_forwards"] == 8.0
    assert arms["latent_flow_n1"]["mean_wm_rollout_forwards"] == 1.0
    assert arms["latent_flow_n64"]["mean_id_forwards"] == 20.0
    assert "WM-LEFLOW-AMORTIZE-001" in T2_METRICS.read_text(encoding="utf-8")


def test_missing_metrics_fail_closed_and_qualification_blocks_confirmation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        _require_arm_schema({"success_rate": 1.0})
    assert "mean_goal_distance" in REQUIRED_ARM_FIELDS
    bad = tmp_path / "qual.json"
    bad.write_text(json.dumps({"status": "WORLD_MODEL_INCOMPETENT", "confirmation_opened": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="qualification"):
        require_qualification_passed(bad)
    config = load_json_config(CONFIG)
    with pytest.raises(RuntimeError, match="closed"):
        run_seed(config, seed=int(config["qualification_seed"]), eval_split="confirmation")


def test_adjudicate_uses_frozen_gates_only() -> None:
    config = load_json_config(CONFIG)
    gates_before = json.dumps(config["gates"], sort_keys=True)

    def _arm(success: float, clock: float, wm: float, distance: float) -> dict:
        return {
            "success_rate": success,
            "mean_wall_clock_s": clock,
            "mean_wm_rollout_forwards": wm,
            "mean_goal_distance": distance,
        }

    fake_row = {
        "seed": 251,
        "horizons": {
            "5": {
                "arms": {
                    "latent_flow_n1": _arm(0.2, 0.01, 1.0, 8.0),
                    "iterative_cem": _arm(0.8, 0.2, 64.0, 3.0),
                    "random_shooting_n64": _arm(0.1, 0.05, 64.0, 9.0),
                }
            }
        },
    }
    metrics = adjudicate([fake_row], config)
    assert json.dumps(config["gates"], sort_keys=True) == gates_before
    assert metrics["status"] == "NEGATIVE_RESULT"
    assert metrics["evidence_level"] == "None"
    assert metrics["does_not_mutate_t2"] is True
    uninformative = json.loads(json.dumps(fake_row))
    uninformative["horizons"]["5"]["arms"]["iterative_cem"]["success_rate"] = 0.1
    uninformative["horizons"]["5"]["arms"]["random_shooting_n64"]["success_rate"] = 0.1
    metrics_u = adjudicate([uninformative], config)
    assert metrics_u["status"] == "UNINFORMATIVE_SUBSTRATE"
    qual = qualify_development(
        {
            "seed": 241,
            "one_step_state_rmse": 9.0,
            "one_step_object_rmse": 9.0,
            "world_model_fingerprint": "abc",
            "horizons": {"5": {"arms": {"random_shooting_n64": {"success_rate": 0.0}}}},
        },
        config,
    )
    assert qual["status"] == "WORLD_MODEL_INCOMPETENT"
    assert qual["confirmation_opened"] is False
