"""Multi-seed goal/dynamics JEPA workspace-candidate study."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.synthetic.multitask_pointmass import (
    DEFAULT_GOALS,
    generate_multitask_pointmass2d,
)
from causal_workspace_jepa.experiments.world_model.manifold_workspace_study import (
    _rollout_audit,
)
from causal_workspace_jepa.interpretability.workspace_tests import (
    conditional_resample_causal_audit,
    counterfactual_subspace_audit,
    discover_shared_subspace,
    local_tangent_subspaces,
    random_subspace,
)
from causal_workspace_jepa.models.deep_jepa import DeepActionConditionedJEPA
from causal_workspace_jepa.models.probes import QuadraticRidgeHead
from causal_workspace_jepa.models.uncertainty import (
    IntervalCalibrator,
    rank_auc,
    spearman_correlation,
)


def run_multitask_workspace_study(config_path: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/cpu_vps.yaml"))
    require_free_disk(resource_profile)
    seeds = [int(value) for value in config.get("seeds", [23, 29, 37])]
    results = [_run_seed(config, seed) for seed in seeds]
    passing_seeds = sum(bool(result["shared_task_workspace_candidate_found"]) for result in results)
    required_seeds = int(config.get("required_passing_seeds", 2))
    shared_candidate = bool(passing_seeds >= required_seeds)
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-T0-005")),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Generalization",
        "model": "goal_and_dynamics_conditioned_deep_numpy_jepa",
        "dataset": "generated_in_memory:MultiTaskPointMass2D",
        "seeds": seeds,
        "runtime_seconds": float(time.monotonic() - started),
        "passing_seed_count": passing_seeds,
        "required_passing_seeds": required_seeds,
        "seed_results": results,
        "shared_task_workspace_candidate_found": shared_candidate,
        "workspace_found": False,
        "missing_workspace_criteria": [
            "reportability_analogue",
            "published_model_replication",
        ],
        "claims": [
            {
                "claim": "Actions are indispensable for held-out goal/dynamics prediction.",
                "evidence_level": "Generalization",
                "supported": bool(sum(result["model_valid"] for result in results) >= required_seeds),
            },
            {
                "claim": "A compact hidden subspace transfers to a held-out goal/dynamics task.",
                "evidence_level": "Generalization",
                "supported": shared_candidate,
            },
            {
                "claim": "A full JEPA workspace was found.",
                "evidence_level": "Generalization",
                "supported": False,
            },
        ],
    }
    output_path = Path(
        str(config.get("output_metrics", "artifacts/metrics/multitask_workspace_study.json"))
    )
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path}",
        resource_profile=resource_profile,
        seed=seeds[0],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output_path.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": str(output_path), "workspace_found": False},
    )
    return metrics


def _run_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    trajectories = int(config.get("trajectories", 192))
    steps = int(config.get("steps", 20))
    heldout_combo = int(config.get("heldout_combination", 7))
    dataset = generate_multitask_pointmass2d(
        trajectories=trajectories,
        steps=steps,
        seed=seed,
    )
    combo_ids = _combination_ids(dataset.observations[:, 0])
    split_ids = _task_splits(combo_ids, heldout_combo=heldout_combo, seed=seed)
    train_obs = dataset.observations[split_ids["predictor_train"]]
    train_actions = dataset.actions[split_ids["predictor_train"]]
    calibration_obs = dataset.observations[split_ids["calibration"]]
    calibration_actions = dataset.actions[split_ids["calibration"]]
    consumer_obs = dataset.observations[split_ids["consumer_train"]]
    consumer_actions = dataset.actions[split_ids["consumer_train"]]
    test_obs = dataset.observations[split_ids["test"]]
    test_actions = dataset.actions[split_ids["test"]]

    models = _fit_ensemble(train_obs, train_actions, config, seed)
    primary = models[0]
    calibration_predictions = _ensemble_physical_predictions(
        models, primary, calibration_obs, calibration_actions
    )
    calibration_targets = calibration_obs[:, 1:, :4].reshape(-1, 4)
    calibrator = IntervalCalibrator.fit(
        calibration_predictions,
        calibration_targets,
        target_coverage=float(config.get("target_coverage", 0.9)),
    )
    consumer_predictions = _ensemble_physical_predictions(
        models, primary, consumer_obs, consumer_actions
    )
    test_predictions = _ensemble_physical_predictions(models, primary, test_obs, test_actions)
    consumer_targets = consumer_obs[:, 1:, :4].reshape(-1, 4)
    test_targets = test_obs[:, 1:, :4].reshape(-1, 4)
    consumer_variance = calibrator.variance(consumer_predictions)
    test_variance = calibrator.variance(test_predictions)

    shuffled_actions = test_actions.reshape(-1, 2).copy()
    np.random.default_rng(seed + 500).shuffle(shuffled_actions, axis=0)
    clean_primary = _physical_predictions(primary, primary, test_obs, test_actions)
    shuffled_primary = _physical_predictions(primary, primary, test_obs, shuffled_actions)
    clean_mse = float(np.mean((clean_primary - test_targets) ** 2))
    shuffled_mse = float(np.mean((shuffled_primary - test_targets) ** 2))
    action_ratio = float(clean_mse / max(shuffled_mse, 1e-12))
    model_valid = bool(action_ratio <= float(config.get("max_action_mse_ratio", 0.5)))

    ood_mass_values = [float(value) for value in config.get("ood_masses", [0.25, 2.4])]
    if len(ood_mass_values) != 2:
        raise ValueError("ood_masses must contain exactly two values")
    ood_masses = (ood_mass_values[0], ood_mass_values[1])
    ood = generate_multitask_pointmass2d(
        trajectories=int(config.get("ood_trajectories", 64)),
        steps=steps,
        seed=seed + 10_000,
        masses=ood_masses,
    )
    ood_predictions = _ensemble_physical_predictions(models, primary, ood.observations, ood.actions)
    ood_targets = ood.observations[:, 1:, :4].reshape(-1, 4)
    id_score = np.mean(consumer_variance, axis=-1)
    ood_score = np.mean(calibrator.variance(ood_predictions), axis=-1)
    id_error = np.mean((consumer_predictions.mean(axis=0) - consumer_targets) ** 2, axis=-1)
    ood_error = np.mean((ood_predictions.mean(axis=0) - ood_targets) ** 2, axis=-1)
    uncertainty_metrics = {
        "calibration": calibrator.evaluate(calibration_predictions, calibration_targets),
        "heldout_combination": calibrator.evaluate(test_predictions, test_targets),
        "ood_auc": rank_auc(id_score, ood_score),
        "error_spearman": spearman_correlation(
            np.concatenate([id_score, ood_score]),
            np.concatenate([id_error, ood_error]),
        ),
    }
    heldout_coverage = uncertainty_metrics["heldout_combination"]["coverage"]
    uncertainty_valid = bool(
        float(config.get("min_heldout_coverage", 0.75))
        <= heldout_coverage
        <= float(config.get("max_heldout_coverage", 0.99))
        and uncertainty_metrics["ood_auc"] >= float(config.get("min_ood_auc", 0.65))
        and uncertainty_metrics["error_spearman"]
        >= float(config.get("min_uncertainty_error_spearman", 0.3))
    )

    train_hidden = _hidden2(primary, consumer_obs, consumer_actions)
    test_hidden = _hidden2(primary, test_obs, test_actions)
    train_consumer_targets = _consumer_targets(
        consumer_obs[:, 1:].reshape(-1, 8),
        consumer_variance,
    )
    test_consumer_targets = _consumer_targets(
        test_obs[:, 1:].reshape(-1, 8),
        test_variance,
    )
    heads = {
        name: QuadraticRidgeHead.fit(
            train_hidden,
            target,
            ridge=float(config.get("head_ridge", 1e-2)),
        )
        for name, target in train_consumer_targets.items()
    }
    head_r2 = {
        name: float(head.score_r2(test_hidden, test_consumer_targets[name]))
        for name, head in heads.items()
    }
    heads_valid = bool(min(head_r2.values()) >= float(config.get("min_consumer_r2", 0.4)))
    jacobian_examples = min(int(config.get("jacobian_examples", 96)), len(train_hidden))
    candidate = discover_shared_subspace(
        {name: head.jacobian(train_hidden[:jacobian_examples]) for name, head in heads.items()},
        max_dimension=int(config.get("max_subspace_dimension", 6)),
        min_consumer_capture=float(config.get("min_consumer_capture", 0.7)),
        max_compactness_ratio=float(config.get("max_compactness_ratio", 0.3)),
        min_consumers=5,
    )
    random_bases = [
        random_subspace(test_hidden.shape[-1], candidate.dimension, seed + 2_000 + index)
        for index in range(int(config.get("random_controls", 32)))
    ]
    tangent_bases = local_tangent_subspaces(
        train_hidden,
        candidate.dimension,
        count=int(config.get("tangent_controls", 32)),
        neighbors=int(config.get("tangent_neighbors", 16)),
        seed=seed + 3_000,
    )
    consumers = {name: head.predict for name, head in heads.items()}
    random_audit = _conditional_audit(
        test_hidden, train_hidden, consumers, candidate.basis, random_bases, config
    )
    tangent_audit = _conditional_audit(
        test_hidden, train_hidden, consumers, candidate.basis, tangent_bases, config
    )

    recipient_obs = test_obs[:, :-1].reshape(-1, 8)
    donor_obs = _counterfactual_context(recipient_obs)
    flat_actions = test_actions.reshape(-1, 2)
    recipient_hidden = primary.hidden_activations(
        primary.encode(recipient_obs).tensor,
        flat_actions,
    )["predictor.hidden2"]
    donor_hidden = primary.hidden_activations(
        primary.encode(donor_obs).tensor,
        flat_actions,
    )["predictor.hidden2"]
    random_counterfactual = _counterfactual_audit(
        recipient_hidden,
        donor_hidden,
        train_hidden,
        consumers,
        candidate.basis,
        random_bases,
        config,
    )
    tangent_counterfactual = _counterfactual_audit(
        recipient_hidden,
        donor_hidden,
        train_hidden,
        consumers,
        candidate.basis,
        tangent_bases,
        config,
    )
    random_rollout = _matched_rollout(
        primary,
        test_obs,
        test_actions,
        train_hidden,
        candidate.basis,
        random_bases,
        random_audit,
        config,
    )
    tangent_rollout = _matched_rollout(
        primary,
        test_obs,
        test_actions,
        train_hidden,
        candidate.basis,
        tangent_bases,
        tangent_audit,
        config,
    )
    minimum_recovery = float(config.get("min_counterfactual_recovery", 0.5))
    minimum_selectivity = float(config.get("min_selective_necessity_ratio", 1.25))
    candidate_found = bool(
        model_valid
        and uncertainty_valid
        and heads_valid
        and candidate.sensitivity_candidate_found
        and random_audit["candidate_density_valid"]
        and tangent_audit["candidate_density_valid"]
        and random_audit["candidate_exceeds_matched_control_p95"]
        and tangent_audit["candidate_exceeds_matched_control_p95"]
        and random_counterfactual["candidate_density_valid"]
        and tangent_counterfactual["candidate_density_valid"]
        and random_counterfactual["candidate_exceeds_matched_control_p95"]
        and tangent_counterfactual["candidate_exceeds_matched_control_p95"]
        and random_counterfactual["candidate"]["mean_recovery"] >= minimum_recovery
        and tangent_counterfactual["candidate"]["mean_recovery"] >= minimum_recovery
        and random_rollout["candidate_exceeds_control_p95"]
        and tangent_rollout["candidate_exceeds_control_p95"]
        and random_rollout["selective_necessity_ratio"] >= minimum_selectivity
        and tangent_rollout["selective_necessity_ratio"] >= minimum_selectivity
    )
    return {
        "seed": seed,
        "split_trajectory_counts": {name: int(len(ids)) for name, ids in split_ids.items()},
        "action_conditioning": {
            "clean_physical_mse": clean_mse,
            "shuffled_action_physical_mse": shuffled_mse,
            "mse_ratio": action_ratio,
        },
        "model_valid": model_valid,
        "uncertainty": uncertainty_metrics,
        "uncertainty_valid": uncertainty_valid,
        "head_test_r2": head_r2,
        "all_consumers_valid": heads_valid,
        "candidate": candidate.to_metrics(),
        "conditional_random_audit": random_audit,
        "conditional_tangent_audit": tangent_audit,
        "counterfactual_random_audit": random_counterfactual,
        "counterfactual_tangent_audit": tangent_counterfactual,
        "random_selective_necessity": random_rollout,
        "tangent_selective_necessity": tangent_rollout,
        "shared_task_workspace_candidate_found": candidate_found,
    }


def _combination_ids(initial_observations: np.ndarray) -> np.ndarray:
    goal_distance = np.sum(
        (initial_observations[:, None, 4:6] - DEFAULT_GOALS[None, :, :]) ** 2,
        axis=-1,
    )
    goal_ids = np.argmin(goal_distance, axis=1)
    mode_ids = np.argmax(initial_observations[:, 6:8], axis=1)
    return (mode_ids * len(DEFAULT_GOALS) + goal_ids).astype(np.int16)


def _task_splits(combo_ids: np.ndarray, *, heldout_combo: int, seed: int) -> dict[str, np.ndarray]:
    seen = np.flatnonzero(combo_ids != heldout_combo)
    heldout = np.flatnonzero(combo_ids == heldout_combo)
    rng = np.random.default_rng(seed + 77)
    rng.shuffle(seen)
    train_end = int(0.7 * len(seen))
    calibration_end = train_end + int(0.15 * len(seen))
    return {
        "predictor_train": np.sort(seen[:train_end]),
        "calibration": np.sort(seen[train_end:calibration_end]),
        "consumer_train": np.sort(seen[calibration_end:]),
        "test": np.sort(heldout),
    }


def _fit_ensemble(
    observations: np.ndarray,
    actions: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> list[DeepActionConditionedJEPA]:
    rng = np.random.default_rng(seed + 101)
    models = []
    shared_encoder = None
    for member in range(int(config.get("ensemble_members", 3))):
        ids = rng.choice(
            len(observations),
            size=max(2, int(len(observations) * float(config.get("bootstrap_fraction", 0.8)))),
            replace=True,
        )
        model = DeepActionConditionedJEPA.fit(
            observations[ids],
            actions[ids],
            latent_dim=int(config.get("latent_dim", 12)),
            hidden_dims=(
                int(config.get("hidden1_dim", 32)),
                int(config.get("hidden2_dim", 24)),
            ),
            learning_rate=float(config.get("learning_rate", 3e-3)),
            training_steps=int(config.get("training_steps", 900)),
            batch_size=int(config.get("batch_size", 128)),
            weight_decay=float(config.get("weight_decay", 1e-5)),
            seed=seed + 131 * member,
            encoder_seed=int(config.get("encoder_seed", 41)),
            encoder=shared_encoder,
        )
        if shared_encoder is None:
            shared_encoder = model.encoder.copy()
        models.append(model)
    return models


def _latent_predictions(
    model: DeepActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    current = model.encode(observations[:, :-1]).tensor.reshape(-1, model.config.latent_dim)
    return model.hidden_activations(current, np.asarray(actions).reshape(-1, 2))[
        "predictor.latent"
    ]


def _physical_predictions(
    model: DeepActionConditionedJEPA,
    decoder_model: DeepActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    latent = _latent_predictions(model, observations, actions)
    design = np.concatenate(
        [latent, np.ones((len(latent), 1), dtype=np.float32)],
        axis=1,
    )
    return (design @ decoder_model.decoder)[:, :4].astype(np.float32)


def _ensemble_physical_predictions(
    models: list[DeepActionConditionedJEPA],
    decoder_model: DeepActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [_physical_predictions(model, decoder_model, observations, actions) for model in models],
        axis=0,
    )


def _hidden2(
    model: DeepActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    current = model.encode(observations[:, :-1]).tensor.reshape(-1, model.config.latent_dim)
    return model.hidden_activations(current, actions.reshape(-1, 2))["predictor.hidden2"]


def _consumer_targets(next_observation: np.ndarray, variance: np.ndarray) -> dict[str, np.ndarray]:
    state = next_observation[:, :4]
    goal = next_observation[:, 4:6]
    mode = np.argmax(next_observation[:, 6:8], axis=1)
    hazards = np.array([[0.0, 0.0], [0.4, -0.4]], dtype=np.float32)[mode]
    offset = state[:, :2] - hazards
    repulsion = 0.12 * offset / np.maximum(np.sum(offset**2, axis=1, keepdims=True), 0.1)
    policy = np.clip(0.8 * (goal - state[:, :2]) - 0.3 * state[:, 2:4] + repulsion, -1, 1)
    return {
        "dynamics_prediction": state.astype(np.float32),
        "value": (-np.sum((state[:, :2] - goal) ** 2, axis=1, keepdims=True)).astype(
            np.float32
        ),
        "risk": (1.0 - np.sum(offset**2, axis=1, keepdims=True)).astype(np.float32),
        "uncertainty": np.log(np.mean(variance, axis=-1, keepdims=True) + 1e-12).astype(
            np.float32
        ),
        "action_selection": policy.astype(np.float32),
    }


def _counterfactual_context(observations: np.ndarray) -> np.ndarray:
    donor = observations.copy()
    donor[:, 4:6] = DEFAULT_GOALS[0]
    donor[:, 6:8] = np.array([1.0, 0.0], dtype=np.float32)
    return donor


def _conditional_audit(
    recipients: np.ndarray,
    bank: np.ndarray,
    consumers: dict[str, Any],
    candidate_basis: np.ndarray,
    controls: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, object]:
    return conditional_resample_causal_audit(
        recipients,
        bank,
        consumers,
        candidate_basis,
        controls,
        neighbor_pool=int(config.get("neighbor_pool", 16)),
        max_density_ratio=float(config.get("max_density_ratio", 3.0)),
        match_factor=float(config.get("control_match_factor", 2.0)),
        min_matched_controls=int(config.get("min_matched_controls", 8)),
    )


def _counterfactual_audit(
    recipients: np.ndarray,
    donors: np.ndarray,
    bank: np.ndarray,
    consumers: dict[str, Any],
    candidate_basis: np.ndarray,
    controls: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, object]:
    return counterfactual_subspace_audit(
        recipients,
        donors,
        bank,
        consumers,
        candidate_basis,
        controls,
        max_density_ratio=float(config.get("max_density_ratio", 3.0)),
        match_factor=float(config.get("control_match_factor", 2.0)),
        min_matched_controls=int(config.get("min_matched_controls", 8)),
    )


def _matched_rollout(
    model: DeepActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
    donor_bank: np.ndarray,
    candidate_basis: np.ndarray,
    controls: list[np.ndarray],
    audit: dict[str, object],
    config: dict[str, Any],
) -> dict[str, Any]:
    indices = [int(index) for index in audit["matched_control_indices"]]
    matched = [controls[index] for index in indices[:16]]
    return _rollout_audit(
        model,
        "predictor.hidden2",
        model.encode(observations[:, 0]).tensor,
        actions,
        donor_bank,
        candidate_basis,
        matched,
        neighbor_pool=int(config.get("neighbor_pool", 16)),
        horizon=int(config.get("rollout_horizon", 8)),
    )
