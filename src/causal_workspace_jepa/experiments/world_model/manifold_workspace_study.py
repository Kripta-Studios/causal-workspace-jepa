"""Deep Tier 0 workspace audit with calibrated uncertainty and manifold controls."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.splits import deterministic_split_ids
from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.interpretability.workspace_tests import (
    conditional_resample_causal_audit,
    conditional_resample_subspace,
    discover_shared_subspace,
    principal_subspace,
    random_subspace,
    selective_necessity_ratio,
)
from causal_workspace_jepa.models.deep_jepa import DeepActionConditionedJEPA
from causal_workspace_jepa.models.probes import QuadraticRidgeHead
from causal_workspace_jepa.models.uncertainty import (
    IntervalCalibrator,
    rank_auc,
    spearman_correlation,
)


def run_manifold_workspace_study(config_path: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/cpu_vps.yaml"))
    require_free_disk(resource_profile)
    seed = int(config.get("seed", 17))
    trajectories = int(config.get("trajectories", 128))
    steps = int(config.get("steps", 24))
    dataset = generate_pointmass2d(trajectories=trajectories, steps=steps, seed=seed)
    splits = deterministic_split_ids(trajectories, seed=seed)
    train_obs = dataset.observations[splits["train"]]
    train_actions = dataset.actions[splits["train"]]
    val_obs, val_actions = (
        dataset.observations[splits["validation"]],
        dataset.actions[splits["validation"]],
    )
    test_obs, test_actions = dataset.observations[splits["test"]], dataset.actions[splits["test"]]

    models = _fit_ensemble(train_obs, train_actions, config, seed)
    primary = models[0]
    val_predictions = _ensemble_predictions(models, val_obs, val_actions)
    test_predictions = _ensemble_predictions(models, test_obs, test_actions)
    val_target_latent = _target_latents(primary, val_obs)
    test_target_latent = _target_latents(primary, test_obs)
    calibrator = IntervalCalibrator.fit(
        val_predictions,
        val_target_latent,
        target_coverage=float(config.get("target_coverage", 0.9)),
    )
    calibration = {
        "scale": calibrator.scale,
        "variance_floor": calibrator.variance_floor,
        "validation": calibrator.evaluate(val_predictions, val_target_latent),
        "test": calibrator.evaluate(test_predictions, test_target_latent),
    }
    val_variance = calibrator.variance(val_predictions)
    test_variance = calibrator.variance(test_predictions)

    ood = generate_pointmass2d(
        trajectories=int(config.get("ood_trajectories", 32)),
        steps=steps,
        seed=seed + 10_000,
        mass=float(config.get("ood_mass", 0.55)),
        drag=float(config.get("ood_drag", 0.2)),
    )
    ood_predictions = _ensemble_predictions(models, ood.observations, ood.actions)
    ood_target_latent = _target_latents(primary, ood.observations)
    ood_variance = calibrator.variance(ood_predictions)
    id_scores = np.mean(test_variance, axis=-1)
    ood_scores = np.mean(ood_variance, axis=-1)
    id_errors = np.mean((test_predictions.mean(axis=0) - test_target_latent) ** 2, axis=-1)
    ood_errors = np.mean((ood_predictions.mean(axis=0) - ood_target_latent) ** 2, axis=-1)
    uncertainty_validity = {
        "ood_auc": rank_auc(id_scores, ood_scores),
        "error_spearman": spearman_correlation(
            np.concatenate([id_scores, ood_scores]),
            np.concatenate([id_errors, ood_errors]),
        ),
        "id_mean_score": float(np.mean(id_scores)),
        "ood_mean_score": float(np.mean(ood_scores)),
        "id_mean_error": float(np.mean(id_errors)),
        "ood_mean_error": float(np.mean(ood_errors)),
    }

    primary_test = test_predictions[0]
    shuffled_actions = test_actions.reshape(-1, test_actions.shape[-1]).copy()
    np.random.default_rng(seed + 900).shuffle(shuffled_actions, axis=0)
    shuffled_predictions = _one_step_predictions(primary, test_obs, shuffled_actions)
    primary_test_mse = float(np.mean((primary_test - test_target_latent) ** 2))
    shuffled_action_mse = float(np.mean((shuffled_predictions - test_target_latent) ** 2))
    model_valid = bool(
        primary_test_mse
        <= float(config.get("max_mse_fraction_of_shuffled", 0.25)) * shuffled_action_mse
    )
    model_metrics = {
        "ensemble_members": len(models),
        "primary_training_loss": primary.training_loss,
        "primary_test_latent_mse": primary_test_mse,
        "ensemble_test_latent_mse": float(
            np.mean((test_predictions.mean(axis=0) - test_target_latent) ** 2)
        ),
        "shuffled_action_test_latent_mse": shuffled_action_mse,
        "action_conditioning_ratio": float(primary_test_mse / max(shuffled_action_mse, 1e-12)),
        "model_valid": model_valid,
    }

    val_sites = _site_activations(primary, val_obs, val_actions)
    test_sites = _site_activations(primary, test_obs, test_actions)
    consumer_targets_val = _consumer_targets(
        val_target_latent,
        val_obs[:, 1:].reshape(-1, val_obs.shape[-1]),
        val_variance,
    )
    consumer_targets_test = _consumer_targets(
        test_target_latent,
        test_obs[:, 1:].reshape(-1, test_obs.shape[-1]),
        test_variance,
    )
    site_results = {}
    for site in ("predictor.hidden1", "predictor.hidden2"):
        site_results[site] = _audit_site(
            model=primary,
            site=site,
            train_activations=val_sites[site],
            test_activations=test_sites[site],
            train_targets=consumer_targets_val,
            test_targets=consumer_targets_test,
            test_initial=primary.encode(test_obs[:, 0]).tensor,
            test_actions=test_actions,
            config=config,
            seed=seed,
        )

    min_coverage = float(config.get("min_test_coverage", 0.8))
    max_coverage = float(config.get("max_test_coverage", 0.98))
    test_calibration = calibration["test"]
    uncertainty_calibrated = bool(
        min_coverage <= test_calibration["coverage"] <= max_coverage
        and test_calibration["gaussian_nll"]
        <= test_calibration["homoscedastic_nll"] + float(config.get("nll_tolerance", 0.05))
        and uncertainty_validity["ood_auc"] >= float(config.get("min_ood_auc", 0.65))
        and uncertainty_validity["error_spearman"]
        >= float(config.get("min_uncertainty_error_spearman", 0.3))
    )
    shared_candidates = [
        site
        for site, result in site_results.items()
        if model_valid and uncertainty_calibrated and result["shared_causal_candidate_found"]
    ]
    workspace_found = False
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-T0-004")),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Specificity",
        "seed": seed,
        "dataset": "generated_in_memory:PointMass2D",
        "model": "deep_numpy_action_conditioned_jepa_ensemble",
        "runtime_seconds": float(time.monotonic() - started),
        "split_trajectory_counts": {name: int(len(ids)) for name, ids in splits.items()},
        "model_metrics": model_metrics,
        "calibrated_uncertainty": calibration,
        "uncertainty_validity": uncertainty_validity,
        "uncertainty_consumer_valid": uncertainty_calibrated,
        "sites": site_results,
        "shared_causal_candidate_sites": shared_candidates,
        "workspace_found": workspace_found,
        "missing_workspace_criteria": [
            "goal_or_instruction_controllability",
            "cross_task_flexible_reuse",
            "reportability_analogue",
        ],
        "claims": [
            {
                "claim": "The learned predictor uses action information on held-out transitions.",
                "evidence_level": "Causal mediation",
                "supported": model_valid,
            },
            {
                "claim": (
                    "Deep-ensemble uncertainty is calibrated and useful for the registered OOD "
                    "shift."
                ),
                "evidence_level": "Generalization",
                "supported": uncertainty_calibrated,
            },
            {
                "claim": (
                    "A compact internal subspace is privileged over manifold-matched controls."
                ),
                "evidence_level": "Specificity",
                "supported": bool(shared_candidates),
            },
            {
                "claim": (
                    "A JEPA workspace satisfying the registered functional criteria was found."
                ),
                "evidence_level": "Specificity",
                "supported": workspace_found,
            },
        ],
    }
    output_path = Path(
        str(config.get("output_metrics", "artifacts/metrics/manifold_workspace_study.json"))
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


def _fit_ensemble(
    observations: np.ndarray,
    actions: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> list[DeepActionConditionedJEPA]:
    members = int(config.get("ensemble_members", 5))
    bootstrap_fraction = float(config.get("bootstrap_fraction", 0.8))
    rng = np.random.default_rng(seed + 100)
    models = []
    shared_encoder = None
    for member in range(members):
        sample_size = max(2, int(round(len(observations) * bootstrap_fraction)))
        ids = rng.choice(len(observations), size=sample_size, replace=True)
        model = DeepActionConditionedJEPA.fit(
            observations[ids],
            actions[ids],
            latent_dim=int(config.get("latent_dim", 8)),
            hidden_dims=(
                int(config.get("hidden1_dim", 24)),
                int(config.get("hidden2_dim", 16)),
            ),
            learning_rate=float(config.get("learning_rate", 3e-3)),
            training_steps=int(config.get("training_steps", 1_200)),
            batch_size=int(config.get("batch_size", 128)),
            weight_decay=float(config.get("weight_decay", 1e-5)),
            seed=seed + member * 101,
            encoder_seed=int(config.get("encoder_seed", 31)),
            encoder=shared_encoder,
        )
        if shared_encoder is None:
            shared_encoder = model.encoder.copy()
        models.append(model)
    return models


def _one_step_predictions(
    model: DeepActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    current = model.encode(observations[:, :-1]).tensor.reshape(-1, model.config.latent_dim)
    action_array = np.asarray(actions, dtype=np.float32).reshape(-1, model.config.action_dim)
    return model.hidden_activations(current, action_array)["predictor.latent"]


def _ensemble_predictions(
    models: list[DeepActionConditionedJEPA],
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [_one_step_predictions(model, observations, actions) for model in models],
        axis=0,
    )


def _target_latents(model: DeepActionConditionedJEPA, observations: np.ndarray) -> np.ndarray:
    return model.encode(observations[:, 1:]).tensor.reshape(-1, model.config.latent_dim)


def _site_activations(
    model: DeepActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
) -> dict[str, np.ndarray]:
    current = model.encode(observations[:, :-1]).tensor.reshape(-1, model.config.latent_dim)
    sites = model.hidden_activations(current, actions.reshape(-1, model.config.action_dim))
    return {name: sites[name] for name in ("predictor.hidden1", "predictor.hidden2")}


def _consumer_targets(
    target_latent: np.ndarray,
    next_state: np.ndarray,
    calibrated_variance: np.ndarray,
) -> dict[str, np.ndarray]:
    goal = np.array([0.75, 0.75], dtype=np.float32)
    policy = np.clip(-0.6 * (next_state[:, :2] - goal) - 0.25 * next_state[:, 2:4], -1, 1)
    return {
        "dynamics_prediction": target_latent.astype(np.float32),
        "value": (-np.sum((next_state[:, :2] - goal) ** 2, axis=1, keepdims=True)).astype(
            np.float32
        ),
        "risk": (
            np.sum(next_state[:, :2] ** 2, axis=1, keepdims=True)
            + 0.25 * np.sum(next_state[:, 2:4] ** 2, axis=1, keepdims=True)
        ).astype(np.float32),
        "uncertainty": np.log(
            np.mean(calibrated_variance, axis=-1, keepdims=True) + 1e-12
        ).astype(np.float32),
        "action_selection": policy.astype(np.float32),
    }


def _audit_site(
    *,
    model: DeepActionConditionedJEPA,
    site: str,
    train_activations: np.ndarray,
    test_activations: np.ndarray,
    train_targets: dict[str, np.ndarray],
    test_targets: dict[str, np.ndarray],
    test_initial: np.ndarray,
    test_actions: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    heads = {
        name: QuadraticRidgeHead.fit(
            train_activations,
            targets,
            ridge=float(config.get("head_ridge", 1e-2)),
        )
        for name, targets in train_targets.items()
    }
    head_r2 = {
        name: float(head.score_r2(test_activations, test_targets[name]))
        for name, head in heads.items()
    }
    jacobian_examples = min(int(config.get("jacobian_examples", 96)), len(train_activations))
    jacobians = {
        name: head.jacobian(train_activations[:jacobian_examples]) for name, head in heads.items()
    }
    representation_dim = train_activations.shape[-1]
    candidate = discover_shared_subspace(
        jacobians,
        max_dimension=min(int(config.get("max_subspace_dimension", 5)), representation_dim),
        min_consumer_capture=float(config.get("min_consumer_capture", 0.75)),
        max_compactness_ratio=float(config.get("max_compactness_ratio", 0.35)),
        min_consumers=5,
    )
    random_bases = [
        random_subspace(representation_dim, candidate.dimension, seed + 2_000 + index)
        for index in range(int(config.get("random_controls", 64)))
    ]
    pca_basis = principal_subspace(train_activations, candidate.dimension)
    consumers = {name: head.predict for name, head in heads.items()}
    causal_audit = conditional_resample_causal_audit(
        test_activations,
        train_activations,
        consumers,
        candidate.basis,
        random_bases,
        pca_basis=pca_basis,
        neighbor_pool=int(config.get("neighbor_pool", 16)),
        max_density_ratio=float(config.get("max_density_ratio", 3.0)),
        match_factor=float(config.get("control_match_factor", 2.0)),
        min_matched_controls=int(config.get("min_matched_controls", 16)),
    )
    matched_indices = list(causal_audit["matched_control_indices"])
    rollout_controls = [random_bases[index] for index in matched_indices[:16]]
    rollout = _rollout_audit(
        model,
        site,
        test_initial,
        test_actions,
        train_activations,
        candidate.basis,
        rollout_controls,
        neighbor_pool=int(config.get("neighbor_pool", 16)),
        horizon=int(config.get("rollout_horizon", 8)),
    )
    overlap = float(
        np.sum((candidate.basis.T @ pca_basis) ** 2) / max(candidate.dimension, 1)
    )
    min_consumer_r2 = float(config.get("min_consumer_r2", 0.5))
    consumers_valid = bool(min(head_r2.values()) >= min_consumer_r2)
    shared_candidate = bool(
        consumers_valid
        and candidate.sensitivity_candidate_found
        and causal_audit["candidate_density_valid"]
        and causal_audit["enough_matched_controls"]
        and causal_audit["candidate_exceeds_matched_control_p95"]
        and causal_audit["pca_control_matched"]
        and causal_audit["candidate_exceeds_pca"]
        and rollout["candidate_exceeds_control_p95"]
        and rollout["selective_necessity_ratio"]
        >= float(config.get("min_selective_necessity_ratio", 1.25))
    )
    return {
        "head_test_r2": head_r2,
        "all_consumers_valid": consumers_valid,
        "candidate": candidate.to_metrics(),
        "candidate_pca_overlap": overlap,
        "conditional_resample_audit": causal_audit,
        "selective_necessity": rollout,
        "shared_causal_candidate_found": shared_candidate,
    }


def _rollout_audit(
    model: DeepActionConditionedJEPA,
    site: str,
    initial: np.ndarray,
    actions: np.ndarray,
    donor_bank: np.ndarray,
    candidate_basis: np.ndarray,
    control_bases: list[np.ndarray],
    *,
    neighbor_pool: int,
    horizon: int,
) -> dict[str, Any]:
    usable_horizon = min(horizon, actions.shape[1])
    clean = _rollout_with_conditional_patch(
        model, site, initial, actions[:, :usable_horizon], donor_bank, None, neighbor_pool
    )
    candidate = _rollout_with_conditional_patch(
        model,
        site,
        initial,
        actions[:, :usable_horizon],
        donor_bank,
        candidate_basis,
        neighbor_pool,
    )
    one_step = _normalized_rollout_damage(clean[:, :1], candidate[:, :1])
    multistep = _normalized_rollout_damage(clean, candidate)
    control_damage = [
        _normalized_rollout_damage(
            clean,
            _rollout_with_conditional_patch(
                model,
                site,
                initial,
                actions[:, :usable_horizon],
                donor_bank,
                basis,
                neighbor_pool,
            ),
        )
        for basis in control_bases
    ]
    control_p95 = float(np.quantile(control_damage, 0.95)) if control_damage else None
    return {
        "one_step_damage": one_step,
        "multistep_damage": multistep,
        "selective_necessity_ratio": selective_necessity_ratio(multistep, one_step),
        "matched_control_count": len(control_damage),
        "matched_control_mean_damage": (
            float(np.mean(control_damage)) if control_damage else None
        ),
        "matched_control_p95_damage": control_p95,
        "candidate_exceeds_control_p95": bool(
            control_damage and control_p95 is not None and multistep > control_p95
        ),
    }


def _rollout_with_conditional_patch(
    model: DeepActionConditionedJEPA,
    site: str,
    initial: np.ndarray,
    actions: np.ndarray,
    donor_bank: np.ndarray,
    basis: np.ndarray | None,
    neighbor_pool: int,
) -> np.ndarray:
    current = np.asarray(initial, dtype=np.float32)
    predictions = []
    for step in range(actions.shape[1]):
        sites = model.hidden_activations(current, actions[:, step])
        hidden = sites[site]
        if basis is not None:
            hidden, _ = conditional_resample_subspace(
                hidden,
                donor_bank,
                basis,
                neighbor_pool=neighbor_pool,
            )
        current = model.predict_from_hidden(site, hidden)
        predictions.append(current)
    return np.stack(predictions, axis=1)


def _normalized_rollout_damage(clean: np.ndarray, intervened: np.ndarray) -> float:
    variance = float(np.mean((clean - clean.mean(axis=0, keepdims=True)) ** 2))
    return float(np.mean((intervened - clean) ** 2) / max(variance, 1e-12))
