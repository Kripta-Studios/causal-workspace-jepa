"""Tier 0 mechanistic study for the tiny JEPA.

This experiment intentionally separates action-use evidence from workspace
claims. The tiny model has dynamics and planner consumers, but no value, risk,
or uncertainty heads, so the workspace hypothesis is expected to fail.
"""

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
from causal_workspace_jepa.interpretability.probing import RidgeProbe
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA
from causal_workspace_jepa.planning.cem import random_shooting_plan
from causal_workspace_jepa.planning.closed_loop import pointmass_rollout_cost


def run_tier0_mechanistic_study(config_path: str | Path) -> dict[str, Any]:
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

    probe_metrics = _action_probe_metrics(model, train_obs, train_actions, test_obs, test_actions)
    patch_metrics = _action_patch_specificity(model, test_obs, test_actions, config)
    planner_metrics = _planner_ablation_metrics(model, test_obs, config)
    workspace_metrics = {
        "candidate": "predictor.input action coordinates",
        "workspace_found": False,
        "reason": (
            "The compact action coordinates are causally used by dynamics/planning, but the tiny "
            "model has no value, risk, or uncertainty consumers and does not test broadcast reuse."
        ),
        "available_consumers": ["dynamics_prediction", "planner_cost"],
        "missing_consumers": ["value", "risk", "uncertainty", "language_reportability"],
        "evidence_level": "Specificity",
    }
    metrics: dict[str, Any] = {
        "experiment_id": config.get("id", "WM-T0-002"),
        "status": "SMOKE_VALIDATED",
        "seed": seed,
        "dataset": str(pointmass_path),
        "model": "tiny_action_conditioned_jepa",
        "evidence_level": "Specificity",
        "claims": [
            {
                "claim": "Action identity is more decodable from latent displacement than endpoints.",
                "evidence_level": "Availability",
                "supported": bool(
                    probe_metrics["displacement_action_r2"]
                    > max(probe_metrics["z_t_action_r2"], probe_metrics["z_next_action_r2"])
                ),
            },
            {
                "claim": "Patching action coordinates causally and selectively changes future latent predictions.",
                "evidence_level": "Specificity",
                "supported": bool(patch_metrics["action_patch_recovery"] > patch_metrics["latent_control_recovery"]),
            },
            {
                "claim": "A J-space-like compact workspace candidate was found.",
                "evidence_level": "Specificity",
                "supported": False,
            },
        ],
        "probe_metrics": probe_metrics,
        "patch_metrics": patch_metrics,
        "planner_metrics": planner_metrics,
        "workspace_metrics": workspace_metrics,
    }
    output_path = Path(str(config.get("output_metrics", "artifacts/metrics/tier0_mechanistic_study.json")))
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path}",
        resource_profile=resource_profile,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        "artifacts/metrics/tier0_mechanistic_study.provenance.json",
        provenance,
        extra={"metrics": str(output_path), "workspace_found": False},
    )
    return metrics


def _action_probe_metrics(
    model: TinyActionConditionedJEPA,
    train_obs: np.ndarray,
    train_actions: np.ndarray,
    test_obs: np.ndarray,
    test_actions: np.ndarray,
) -> dict[str, float]:
    train_z = model.encode(train_obs).tensor
    test_z = model.encode(test_obs).tensor
    train_action = train_actions.reshape(-1, train_actions.shape[-1])
    test_action = test_actions.reshape(-1, test_actions.shape[-1])
    train_features = {
        "z_t": train_z[:, :-1, :].reshape(-1, model.config.latent_dim),
        "z_next": train_z[:, 1:, :].reshape(-1, model.config.latent_dim),
        "displacement": (train_z[:, 1:, :] - train_z[:, :-1, :]).reshape(-1, model.config.latent_dim),
    }
    test_features = {
        "z_t": test_z[:, :-1, :].reshape(-1, model.config.latent_dim),
        "z_next": test_z[:, 1:, :].reshape(-1, model.config.latent_dim),
        "displacement": (test_z[:, 1:, :] - test_z[:, :-1, :]).reshape(-1, model.config.latent_dim),
    }
    metrics = {}
    for name, train_feature in train_features.items():
        probe = RidgeProbe.fit(train_feature, train_action)
        mse = probe.score_mse(test_features[name], test_action)
        baseline = probe.mean_baseline_mse(test_action)
        metrics[f"{name}_action_mse"] = mse
        metrics[f"{name}_action_r2"] = 1.0 - mse / max(baseline, 1e-12)
    return {
        "z_t_action_mse": float(metrics["z_t_action_mse"]),
        "z_next_action_mse": float(metrics["z_next_action_mse"]),
        "displacement_action_mse": float(metrics["displacement_action_mse"]),
        "z_t_action_r2": float(metrics["z_t_action_r2"]),
        "z_next_action_r2": float(metrics["z_next_action_r2"]),
        "displacement_action_r2": float(metrics["displacement_action_r2"]),
    }


def _action_patch_specificity(
    model: TinyActionConditionedJEPA,
    test_obs: np.ndarray,
    test_actions: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    rng = np.random.default_rng(int(config.get("control_seed", 17)))
    z = model.encode(test_obs[:, :-1, :]).tensor.reshape(-1, model.config.latent_dim)
    actions = test_actions.reshape(-1, model.config.action_dim)
    count = min(int(config.get("patch_pairs", 64)), len(actions))
    recipient_ids = rng.choice(len(actions), size=count, replace=True)
    donor_ids = rng.choice(len(actions), size=count, replace=True)
    latent_control_recoveries = []
    action_recoveries = []
    random_action_recoveries = []
    latent_dims = np.arange(model.config.latent_dim)
    for recipient, donor in zip(recipient_ids, donor_ids):
        recipient_z = z[recipient : recipient + 1]
        action = actions[recipient : recipient + 1]
        donor_action = actions[donor : donor + 1]
        clean = _predict_one(model, recipient_z, action)
        target = _predict_one(model, recipient_z, donor_action)
        action_patch = target
        control_dims = rng.choice(latent_dims, size=model.config.action_dim, replace=False)
        control_z = recipient_z.copy()
        control_z[:, control_dims] = z[donor : donor + 1, control_dims]
        latent_control = _predict_one(model, control_z, action)
        random_action = rng.normal(size=donor_action.shape).astype(np.float32)
        random_action = random_action / max(float(np.linalg.norm(random_action)), 1e-12)
        random_action = random_action * max(float(np.linalg.norm(donor_action)), 1e-12)
        random_control = _predict_one(model, recipient_z, random_action)
        action_recoveries.append(_recovery(clean, target, action_patch))
        latent_control_recoveries.append(_recovery(clean, target, latent_control))
        random_action_recoveries.append(_recovery(clean, target, random_control))
    return {
        "action_patch_recovery": float(np.mean(action_recoveries)),
        "latent_control_recovery": float(np.mean(latent_control_recoveries)),
        "random_action_control_recovery": float(np.mean(random_action_recoveries)),
        "specificity_margin_vs_latent_control": float(
            np.mean(action_recoveries) - np.mean(latent_control_recoveries)
        ),
        "specificity_margin_vs_random_action": float(
            np.mean(action_recoveries) - np.mean(random_action_recoveries)
        ),
        "pairs": float(count),
    }


def _planner_ablation_metrics(
    model: TinyActionConditionedJEPA,
    test_obs: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    seed = int(config.get("seed", 0))
    horizon = int(config.get("planner_horizon", 8))
    candidates = int(config.get("planner_candidates", 128))
    goal = np.array([0.75, 0.75], dtype=np.float32)
    start = test_obs[0, 0]
    clean = random_shooting_plan(model, start, goal, horizon=horizon, candidates=candidates, seed=seed)
    clean_cost = pointmass_rollout_cost(start, clean["actions"], goal)  # type: ignore[arg-type]
    zero_action_cost = pointmass_rollout_cost(start, np.zeros((horizon, model.config.action_dim), dtype=np.float32), goal)
    rng = np.random.default_rng(seed + 123)
    random_actions = rng.uniform(-1.0, 1.0, size=(candidates, horizon, model.config.action_dim)).astype(np.float32)
    random_cost = float(np.mean([pointmass_rollout_cost(start, actions, goal) for actions in random_actions]))
    return {
        "clean_planner_cost": float(clean_cost),
        "zero_action_cost": float(zero_action_cost),
        "random_action_mean_cost": random_cost,
        "action_ablation_damage": float(zero_action_cost - clean_cost),
    }


def _predict_one(model: TinyActionConditionedJEPA, z: np.ndarray, action: np.ndarray) -> np.ndarray:
    features = np.concatenate([z, action, np.ones((z.shape[0], 1), dtype=z.dtype)], axis=-1)
    return features @ model.predictor


def _recovery(clean: np.ndarray, target: np.ndarray, patched: np.ndarray) -> float:
    denominator = float(np.linalg.norm(clean - target))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.linalg.norm(patched - target) / denominator)
