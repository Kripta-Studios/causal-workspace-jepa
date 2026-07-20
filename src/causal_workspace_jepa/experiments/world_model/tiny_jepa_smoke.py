"""End-to-end tiny JEPA smoke experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.splits import deterministic_split_ids
from causal_workspace_jepa.data.synthetic.base import load_dataset
from causal_workspace_jepa.data.synthetic.generate import generate_tier0
from causal_workspace_jepa.data.synthetic.pointmass import step_pointmass
from causal_workspace_jepa.models.tiny_jepa import (
    TinyActionConditionedJEPA,
    evaluate_latent_mse,
    mean_latent_baseline_mse,
)
from causal_workspace_jepa.planning.cem import random_shooting_plan
from causal_workspace_jepa.planning.closed_loop import pointmass_rollout_cost


def run_tiny_jepa_smoke(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/cpu_vps.yaml"))
    require_free_disk(resource_profile)
    seed = int(config.get("seed", 0))
    latent_dim = int(config.get("latent_dim", 16))
    dataset_dir = Path(str(config.get("dataset", "data/processed/tier0_smoke")))
    pointmass_path = dataset_dir / "pointmass2d.npz"
    if not pointmass_path.exists():
        generate_tier0("configs/data/tier0_smoke.yaml")
    dataset = load_dataset(pointmass_path)
    splits = deterministic_split_ids(dataset.observations.shape[0], seed=seed)
    train_obs = dataset.observations[splits["train"]]
    train_actions = dataset.actions[splits["train"]]
    test_obs = dataset.observations[splits["test"]]
    test_actions = dataset.actions[splits["test"]]

    model = TinyActionConditionedJEPA.fit(
        train_obs,
        train_actions,
        latent_dim=latent_dim,
        seed=seed,
        action_mode="conditioned",
    )
    no_action = TinyActionConditionedJEPA.fit(
        train_obs,
        train_actions,
        latent_dim=latent_dim,
        seed=seed,
        action_mode="no_action",
    )
    shuffled = TinyActionConditionedJEPA.fit(
        train_obs,
        train_actions,
        latent_dim=latent_dim,
        seed=seed,
        action_mode="shuffled_action",
    )

    conditioned_mse = evaluate_latent_mse(model, test_obs, test_actions)
    no_action_mse = evaluate_latent_mse(no_action, test_obs, test_actions)
    shuffled_mse = evaluate_latent_mse(shuffled, test_obs, test_actions)
    mean_mse = mean_latent_baseline_mse(model, test_obs)

    checkpoint_path = Path("artifacts/checkpoints/tiny_jepa_smoke.npz")
    model.save(checkpoint_path)
    reloaded = TinyActionConditionedJEPA.load(checkpoint_path)
    reload_mse = evaluate_latent_mse(reloaded, test_obs, test_actions)

    goal = np.array([0.75, 0.75], dtype=np.float32)
    start = test_obs[0, 0]
    plan = random_shooting_plan(
        model,
        start,
        goal,
        horizon=8,
        candidates=128,
        seed=seed,
    )
    planned_cost = pointmass_rollout_cost(start, plan["actions"], goal)  # type: ignore[arg-type]
    rng = np.random.default_rng(seed + 99)
    random_actions = rng.uniform(-1.0, 1.0, size=(128, 8, 2)).astype(np.float32)
    random_costs = np.array([pointmass_rollout_cost(start, actions, goal) for actions in random_actions])
    random_rollout_cost = float(random_costs.mean())
    first_step_state = step_pointmass(start, plan["first_action"])  # type: ignore[arg-type]

    metrics = {
        "experiment_id": config.get("id", "WM-T0-001"),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Availability",
        "seed": seed,
        "dataset": str(pointmass_path),
        "checkpoint": str(checkpoint_path),
        "latent_dim": latent_dim,
        "test_trajectories": int(test_obs.shape[0]),
        "conditioned_latent_mse": conditioned_mse,
        "no_action_latent_mse": no_action_mse,
        "shuffled_action_latent_mse": shuffled_mse,
        "mean_baseline_latent_mse": mean_mse,
        "reload_latent_mse": reload_mse,
        "planner_predicted_cost": float(plan["predicted_cost"]),
        "planner_true_cost": planned_cost,
        "random_rollout_mean_cost": random_rollout_cost,
        "planner_first_action": np.asarray(plan["first_action"]).tolist(),
        "first_step_position": first_step_state[:2].tolist(),
        "passes": {
            "beats_mean_baseline": conditioned_mse < mean_mse,
            "beats_no_action": conditioned_mse < no_action_mse,
            "beats_shuffled_action": conditioned_mse <= shuffled_mse * 1.05,
            "reload_matches": abs(reload_mse - conditioned_mse) < 1e-9,
            "planner_beats_random": planned_cost < random_rollout_cost,
        },
    }
    metrics["all_passed"] = bool(all(metrics["passes"].values()))
    output_path = Path(str(config.get("output_metrics", "artifacts/metrics/tiny_jepa_smoke.json")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        "artifacts/metrics/tiny_jepa_smoke.provenance.json",
        collect_provenance(
            command=f"python scripts/run_experiment.py --config {config_path}",
            resource_profile=resource_profile,
            seed=seed,
        ),
        extra={"metrics": str(output_path), "all_passed": metrics["all_passed"]},
    )
    if not metrics["all_passed"]:
        raise RuntimeError(f"tiny JEPA smoke failed acceptance checks: {metrics['passes']}")
    return metrics
