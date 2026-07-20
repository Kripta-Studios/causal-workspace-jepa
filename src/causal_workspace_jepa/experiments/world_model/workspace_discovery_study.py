"""Multi-consumer Tier 0 workspace discovery with falsifying controls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.splits import deterministic_split_ids
from causal_workspace_jepa.data.synthetic.base import load_dataset
from causal_workspace_jepa.data.synthetic.generate import generate_tier0
from causal_workspace_jepa.interpretability.workspace_tests import (
    discover_shared_subspace,
    matched_causal_subspace_audit,
    principal_subspace,
    project_out_subspace,
    random_subspace,
    selective_necessity_ratio,
)
from causal_workspace_jepa.models.probes import QuadraticRidgeHead
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA


def run_workspace_discovery_study(config_path: str | Path) -> dict[str, Any]:
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

    train_representation = _predicted_next_latents(model, train_obs, train_actions)
    test_representation = _predicted_next_latents(model, test_obs, test_actions)
    train_state = train_obs[:, 1:, :].reshape(-1, train_obs.shape[-1])
    test_state = test_obs[:, 1:, :].reshape(-1, test_obs.shape[-1])
    targets = _consumer_targets(train_state, test_state)
    heads = {
        name: QuadraticRidgeHead.fit(
            train_representation,
            pair[0],
            ridge=float(config.get("head_ridge", 1e-3)),
        )
        for name, pair in targets.items()
    }
    head_r2 = {
        name: float(head.score_r2(test_representation, targets[name][1]))
        for name, head in heads.items()
    }

    jacobian_examples = min(int(config.get("jacobian_examples", 64)), len(test_representation))
    audit_representation = test_representation[:jacobian_examples]
    jacobians = {name: head.jacobian(audit_representation) for name, head in heads.items()}
    max_dimension = int(config.get("max_subspace_dimension", 4))
    min_capture = float(config.get("min_consumer_capture", 0.75))
    max_compactness = float(config.get("max_compactness_ratio", 0.3))
    candidate = discover_shared_subspace(
        jacobians,
        max_dimension=max_dimension,
        min_consumer_capture=min_capture,
        max_compactness_ratio=max_compactness,
        min_consumers=5,
    )

    consumers = {name: head.predict for name, head in heads.items()}
    control_count = int(config.get("random_controls", 32))
    random_bases = [
        random_subspace(latent_dim, candidate.dimension, seed + 1000 + index)
        for index in range(control_count)
    ]
    center = train_representation.mean(axis=0)
    causal_audit = matched_causal_subspace_audit(
        audit_representation,
        consumers,
        candidate.basis,
        random_bases,
        center=center,
    )
    pca_basis = principal_subspace(train_representation, candidate.dimension)
    pca_audit = matched_causal_subspace_audit(
        audit_representation,
        consumers,
        pca_basis,
        random_bases,
        center=center,
    )
    causal_audit["pca_control_mean_damage"] = pca_audit["candidate_mean_damage"]
    causal_audit["specificity_margin_vs_pca"] = float(
        causal_audit["candidate_mean_damage"] - pca_audit["candidate_mean_damage"]
    )

    detector_controls = _detector_controls(
        representation_dim=latent_dim,
        max_dimension=max_dimension,
        min_capture=min_capture,
        max_compactness=max_compactness,
        seed=seed + 700,
    )
    necessity = _rollout_necessity(
        model,
        test_obs,
        test_actions,
        candidate.basis,
        random_bases,
        center,
        horizon=int(config.get("rollout_horizon", 8)),
    )

    all_heads_predictive = bool(
        min(head_r2.values()) >= float(config.get("min_consumer_r2", 0.8))
    )
    shared_candidate = bool(
        candidate.sensitivity_candidate_found
        and all_heads_predictive
        and causal_audit["specificity_exceeds_all_random_controls"]
    )
    pca_specific = bool(float(causal_audit["specificity_margin_vs_pca"]) > 0.05)
    workspace_found = False
    missing_criteria = [
        "goal_or_instruction_controllability",
        "depth_and_horizon_evolution",
        "held_out_task_generalization",
    ]
    if not pca_specific:
        missing_criteria.append("specificity_over_high_variance_pca")

    metrics: dict[str, Any] = {
        "experiment_id": config.get("id", "WM-T0-003"),
        "status": "SMOKE_VALIDATED",
        "seed": seed,
        "dataset": str(pointmass_path),
        "model": "tiny_action_conditioned_jepa_with_frozen_multiconsumer_heads",
        "evidence_level": "Specificity",
        "head_test_r2": head_r2,
        "candidate": candidate.to_metrics(),
        "causal_ablation": causal_audit,
        "selective_necessity": necessity,
        "detector_controls": detector_controls,
        "shared_causal_subspace_candidate_found": shared_candidate,
        "workspace_found": workspace_found,
        "missing_workspace_criteria": missing_criteria,
        "claims": [
            {
                "claim": (
                    "The detector recovers a known shared subspace and rejects disjoint consumers."
                ),
                "evidence_level": "Availability",
                "supported": bool(
                    detector_controls["positive_control_found"]
                    and not detector_controls["negative_control_found"]
                ),
            },
            {
                "claim": "A compact subspace is jointly used by five frozen JEPA consumers.",
                "evidence_level": "Specificity",
                "supported": shared_candidate,
            },
            {
                "claim": (
                    "The candidate is more causally privileged than a high-variance PCA subspace."
                ),
                "evidence_level": "Specificity",
                "supported": pca_specific,
            },
            {
                "claim": "A JEPA workspace satisfying the preregistered criteria was found.",
                "evidence_level": "Specificity",
                "supported": workspace_found,
            },
        ],
    }
    output_path = Path(
        str(config.get("output_metrics", "artifacts/metrics/workspace_discovery_study.json"))
    )
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path}",
        resource_profile=resource_profile,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output_path.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": str(output_path), "workspace_found": workspace_found},
    )
    return metrics


def _predicted_next_latents(
    model: TinyActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    z_t = model.encode(observations[:, :-1, :]).tensor.reshape(-1, model.config.latent_dim)
    flat_actions = actions.reshape(-1, model.config.action_dim)
    features = np.concatenate(
        [z_t, flat_actions, np.ones((len(z_t), 1), dtype=np.float32)],
        axis=-1,
    )
    return (features @ model.predictor).astype(np.float32)


def _consumer_targets(
    train_state: np.ndarray,
    test_state: np.ndarray,
) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
    goal = np.array([0.75, 0.75], dtype=np.float32)
    mean = train_state.mean(axis=0)
    covariance = np.cov(train_state.T) + 1e-3 * np.eye(train_state.shape[-1])
    precision = np.linalg.inv(covariance)

    def build(state: np.ndarray) -> dict[str, np.ndarray]:
        centered = state - mean
        uncertainty = np.einsum("ni,ij,nj->n", centered, precision, centered)
        policy = -0.5 * (state[:, :2] - goal) - 0.2 * state[:, 2:4]
        return {
            "dynamics_prediction": state.astype(np.float32),
            "value": (-np.sum((state[:, :2] - goal) ** 2, axis=1, keepdims=True)).astype(
                np.float32
            ),
            "risk": np.sum(state[:, :2] ** 2, axis=1, keepdims=True).astype(np.float32),
            "uncertainty": np.log1p(uncertainty[:, None]).astype(np.float32),
            "action_selection": policy.astype(np.float32),
        }

    train_targets = build(train_state)
    test_targets = build(test_state)
    return {name: (target, test_targets[name]) for name, target in train_targets.items()}


def _detector_controls(
    *,
    representation_dim: int,
    max_dimension: int,
    min_capture: float,
    max_compactness: float,
    seed: int,
) -> dict[str, object]:
    names = ["dynamics_prediction", "value", "risk", "uncertainty", "action_selection"]
    shared_dimension = min(3, max_dimension)
    shared_basis = random_subspace(representation_dim, shared_dimension, seed)
    rng = np.random.default_rng(seed + 1)
    positive = {
        name: rng.normal(size=(3, shared_dimension)) @ shared_basis.T for name in names
    }
    private_dimension = 2
    negative = {}
    for index, name in enumerate(names):
        basis = np.zeros((representation_dim, private_dimension), dtype=np.float64)
        start = index * private_dimension
        basis[start : start + private_dimension] = np.eye(private_dimension)
        negative[name] = rng.normal(size=(3, private_dimension)) @ basis.T
    positive_result = discover_shared_subspace(
        positive,
        max_dimension=max_dimension,
        min_consumer_capture=min_capture,
        max_compactness_ratio=max_compactness,
        min_consumers=5,
    )
    negative_result = discover_shared_subspace(
        negative,
        max_dimension=max_dimension,
        min_consumer_capture=min_capture,
        max_compactness_ratio=max_compactness,
        min_consumers=5,
    )
    return {
        "positive_control_found": positive_result.sensitivity_candidate_found,
        "positive_control": positive_result.to_metrics(),
        "negative_control_found": negative_result.sensitivity_candidate_found,
        "negative_control": negative_result.to_metrics(),
    }


def _rollout_necessity(
    model: TinyActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
    candidate_basis: np.ndarray,
    random_bases: list[np.ndarray],
    center: np.ndarray,
    *,
    horizon: int,
) -> dict[str, float]:
    usable_horizon = min(horizon, actions.shape[1])
    initial = model.encode(observations[:, 0, :]).tensor
    clean = _rollout_with_ablation(model, initial, actions[:, :usable_horizon], None, center)
    candidate = _rollout_with_ablation(
        model,
        initial,
        actions[:, :usable_horizon],
        candidate_basis,
        center,
    )
    one_step_damage = _normalized_rollout_damage(clean[:, :1], candidate[:, :1])
    multistep_damage = _normalized_rollout_damage(clean, candidate)
    random_multistep = [
        _normalized_rollout_damage(
            clean,
            _rollout_with_ablation(model, initial, actions[:, :usable_horizon], basis, center),
        )
        for basis in random_bases
    ]
    return {
        "one_step_damage": one_step_damage,
        "multistep_damage": multistep_damage,
        "selective_necessity_ratio": selective_necessity_ratio(multistep_damage, one_step_damage),
        "random_control_multistep_mean_damage": float(np.mean(random_multistep)),
        "random_control_multistep_max_damage": float(np.max(random_multistep)),
        "specificity_margin_vs_random_mean": float(multistep_damage - np.mean(random_multistep)),
    }


def _rollout_with_ablation(
    model: TinyActionConditionedJEPA,
    initial: np.ndarray,
    actions: np.ndarray,
    basis: np.ndarray | None,
    center: np.ndarray,
) -> np.ndarray:
    current = initial.astype(np.float32)
    predictions = []
    for step in range(actions.shape[1]):
        if basis is not None:
            current = project_out_subspace(current, basis, center=center)
        features = np.concatenate(
            [current, actions[:, step, :], np.ones((len(current), 1), dtype=np.float32)],
            axis=-1,
        )
        current = features @ model.predictor
        predictions.append(current)
    return np.stack(predictions, axis=1)


def _normalized_rollout_damage(clean: np.ndarray, ablated: np.ndarray) -> float:
    variance = float(np.mean((clean - clean.mean(axis=0, keepdims=True)) ** 2))
    return float(np.mean((ablated - clean) ** 2) / max(variance, 1e-12))
