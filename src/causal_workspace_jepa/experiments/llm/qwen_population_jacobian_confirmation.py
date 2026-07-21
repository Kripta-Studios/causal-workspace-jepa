"""Held-out confirmation of population-averaged Qwen Jacobian transport."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.activation_store import read_hdf5_shards
from causal_workspace_jepa.experiments.llm.qwen_context_geometry_study import (
    _entity_split_ids,
)
from causal_workspace_jepa.models.intervention_jepa import effect_correlation


def run_qwen_population_jacobian_confirmation(config_path: str | Path) -> dict[str, Any]:
    """Confirm or reject the post-v1 population-Jacobian observation on validation entities."""

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    hardware = require_free_disk(resource_profile)
    seed = int(config.get("seed", 353))
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path.as_posix()}",
        resource_profile=resource_profile,
        seed=seed,
    )
    if provenance.git_dirty:
        raise RuntimeError("LLM-POPULATION-JACOBIAN-001 requires a clean committed worktree")
    patch, patch_records = read_hdf5_shards(str(config["patch_dataset_dir"]))
    geometry, geometry_records = read_hdf5_shards(str(config["geometry_dataset_dir"]))
    jacobians = geometry["logit_jacobian"].astype(np.float64)
    entity_splits = _entity_split_ids()
    train_entities = np.flatnonzero(entity_splits == 0)
    validation_entities = np.flatnonzero(entity_splits == 1)
    validation = patch["split_id"] == 1
    answer_count = patch["clean_answer_logits"].shape[1]
    directions = patch["source_delta"][validation].astype(np.float64)
    observed = patch["target_effect"][validation, -answer_count:].astype(np.float64)
    recipients = patch["recipient_id"][validation].astype(np.int64)
    local_prediction = np.stack(
        [jacobians[recipient] @ direction for recipient, direction in zip(recipients, directions, strict=True)]
    )
    population_jacobian = jacobians[train_entities].mean(axis=0)
    population_prediction = directions @ population_jacobian.T
    quadratic_prediction = patch["quadratic_taylor"][validation, -answer_count:].astype(
        np.float64
    )
    local_score = _scores(local_prediction, observed, patch, validation)
    population_score = _scores(population_prediction, observed, patch, validation)
    quadratic_score = _scores(quadratic_prediction, observed, patch, validation)
    per_context = _per_context_comparison(
        recipients,
        validation_entities,
        local_prediction,
        population_prediction,
        observed,
    )
    rng = np.random.default_rng(seed)
    curve = _averaging_curve(
        config,
        rng,
        jacobians,
        train_entities,
        directions,
        observed,
        patch,
        validation,
    )
    row_null = _answer_row_permutation_null(
        config,
        rng,
        population_jacobian,
        directions,
        observed,
        patch,
        validation,
    )
    bootstrap = _paired_bootstrap(
        local_prediction,
        population_prediction,
        observed,
        draws=int(config.get("bootstrap_draws", 10000)),
        rng=np.random.default_rng(int(config.get("bootstrap_seed", 359))),
    )
    population_decision = bool(
        population_score["normalized_mse"]
        <= local_score["normalized_mse"] * float(config.get("local_mse_ratio_max", 0.80))
        and population_score["correlation"]
        >= local_score["correlation"] + float(config.get("correlation_margin_min", 0.03))
        and population_score["answer_candidate_agreement"]
        >= local_score["answer_candidate_agreement"]
        + float(config.get("candidate_margin_min", 0.10))
        and per_context["population_better_mse_contexts"]
        >= int(config.get("contexts_improved_min", 4))
    )
    size_one = curve["1"]
    size_sixteen = curve["16"]
    size_twenty_four = curve["24"]
    median_mse_by_size = np.asarray(
        [curve[str(size)]["median_normalized_mse"] for size in (1, 2, 4, 8, 16, 24)]
    )
    log_size_correlation = float(
        np.corrcoef(np.log(np.asarray([1, 2, 4, 8, 16, 24])), median_mse_by_size)[0, 1]
    )
    averaging_decision = bool(
        size_sixteen["median_normalized_mse"]
        <= size_one["median_normalized_mse"]
        * float(config.get("size_curve_mse_ratio_max", 0.80))
        and log_size_correlation
        <= float(config.get("size_curve_log_correlation_max", -0.80))
        and size_twenty_four["median_candidate_agreement"]
        >= size_one["median_candidate_agreement"]
        + float(config.get("size_curve_candidate_margin_min", 0.10))
    )
    semantic_control_decision = bool(
        population_score["normalized_mse"]
        <= row_null["p05_normalized_mse"]
        * float(config.get("row_null_mse_ratio_max", 0.80))
        and population_score["answer_candidate_agreement"]
        >= row_null["p95_candidate_agreement"]
        + float(config.get("row_null_candidate_margin_min", 0.05))
    )
    decisions = {
        "h_geo_04_population_jacobian_confirmation": population_decision,
        "h_geo_05_averaging_regularization_curve": averaging_decision,
        "h_geo_06_answer_semantic_specificity": semantic_control_decision,
    }
    discovery_metrics = json.loads(
        Path(str(config["discovery_metrics"])).read_text(encoding="utf-8")
    )["geometry"]["transport"]
    status = (
        "COMPLETED_POSITIVE"
        if all(decisions.values())
        else "COMPLETED_MIXED"
        if any(decisions.values())
        else "COMPLETED_NEGATIVE"
    )
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "LLM-POPULATION-JACOBIAN-001")),
        "status": status,
        "evidence_level": "Generalization",
        "dataset_id": "LLM-CAPITAL-PATCH-001 plus LLM-CONTEXT-GEOMETRY-001",
        "patch_examples": len(patch_records),
        "jacobian_contexts": len(geometry_records),
        "confirmation_split": "validation only; not used in v1 geometry decisions",
        "validation_entities": validation_entities.tolist(),
        "validation_examples": int(validation.sum()),
        "train_jacobian_contexts": len(train_entities),
        "scores": {
            "matched_local_jacobian": local_score,
            "train_population_jacobian": population_score,
            "quadratic_taylor": quadratic_score,
        },
        "per_context": per_context,
        "averaging_curve": curve,
        "averaging_curve_log_size_mse_correlation": log_size_correlation,
        "answer_row_permutation_null": row_null,
        "paired_bootstrap": bootstrap,
        "hypothesis_decisions": decisions,
        "discovery_test_reference": discovery_metrics,
        "hardware": hardware.as_dict(),
        "runtime_seconds": float(time.perf_counter() - started),
        "scientific_boundary": (
            "This confirmation tests population averaging of local logit Jacobians for finite donor "
            "chords. It does not identify a circuit, semantic activation basis, or workspace. The "
            "thresholds were chosen after the test-split discovery and are valid only on this "
            "previously unanalyzed validation split."
        ),
    }
    output = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen_population_jacobian_v1.json"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": output.as_posix(), "hypothesis_decisions": decisions},
    )
    return metrics


def _averaging_curve(
    config: dict[str, Any],
    rng: np.random.Generator,
    jacobians: np.ndarray,
    train_entities: np.ndarray,
    directions: np.ndarray,
    observed: np.ndarray,
    patch: dict[str, np.ndarray],
    validation: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    draws = int(config.get("subset_draws", 128))
    result: dict[str, dict[str, float | int]] = {}
    for size in (1, 2, 4, 8, 16, 24):
        scores = []
        actual_draws = 1 if size == len(train_entities) else draws
        for _ in range(actual_draws):
            subset = rng.choice(train_entities, size=size, replace=False)
            prediction = directions @ jacobians[subset].mean(axis=0).T
            scores.append(_scores(prediction, observed, patch, validation))
        mse = np.asarray([score["normalized_mse"] for score in scores])
        candidate = np.asarray(
            [score["answer_candidate_agreement"] for score in scores]
        )
        result[str(size)] = {
            "draws": actual_draws,
            "median_normalized_mse": float(np.median(mse)),
            "p05_normalized_mse": float(np.quantile(mse, 0.05)),
            "p95_normalized_mse": float(np.quantile(mse, 0.95)),
            "median_candidate_agreement": float(np.median(candidate)),
            "p05_candidate_agreement": float(np.quantile(candidate, 0.05)),
            "p95_candidate_agreement": float(np.quantile(candidate, 0.95)),
        }
    return result


def _answer_row_permutation_null(
    config: dict[str, Any],
    rng: np.random.Generator,
    population_jacobian: np.ndarray,
    directions: np.ndarray,
    observed: np.ndarray,
    patch: dict[str, np.ndarray],
    validation: np.ndarray,
) -> dict[str, float | int]:
    scores = []
    for _ in range(int(config.get("row_permutations", 256))):
        permuted = population_jacobian[rng.permutation(population_jacobian.shape[0])]
        scores.append(_scores(directions @ permuted.T, observed, patch, validation))
    mse = np.asarray([score["normalized_mse"] for score in scores])
    candidate = np.asarray([score["answer_candidate_agreement"] for score in scores])
    return {
        "permutations": len(scores),
        "mean_normalized_mse": float(mse.mean()),
        "p05_normalized_mse": float(np.quantile(mse, 0.05)),
        "p95_normalized_mse": float(np.quantile(mse, 0.95)),
        "mean_candidate_agreement": float(candidate.mean()),
        "p05_candidate_agreement": float(np.quantile(candidate, 0.05)),
        "p95_candidate_agreement": float(np.quantile(candidate, 0.95)),
    }


def _per_context_comparison(
    recipients: np.ndarray,
    validation_entities: np.ndarray,
    local: np.ndarray,
    population: np.ndarray,
    observed: np.ndarray,
) -> dict[str, Any]:
    rows = []
    better = 0
    for entity in validation_entities:
        mask = recipients == entity
        power = max(float(np.mean(observed[mask] ** 2)), 1e-12)
        local_mse = float(np.mean((local[mask] - observed[mask]) ** 2) / power)
        population_mse = float(np.mean((population[mask] - observed[mask]) ** 2) / power)
        better += int(population_mse < local_mse)
        rows.append(
            {
                "entity_id": int(entity),
                "local_normalized_mse": local_mse,
                "population_normalized_mse": population_mse,
                "population_better": population_mse < local_mse,
            }
        )
    return {"population_better_mse_contexts": better, "contexts": rows}


def _paired_bootstrap(
    local: np.ndarray,
    population: np.ndarray,
    observed: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    local_error = np.mean((local - observed) ** 2, axis=1)
    population_error = np.mean((population - observed) ** 2, axis=1)
    improvement = local_error - population_error
    indices = rng.integers(0, len(improvement), size=(draws, len(improvement)))
    bootstrapped = improvement[indices].mean(axis=1)
    return {
        "draws": draws,
        "mean_raw_mse_improvement": float(improvement.mean()),
        "ci95_low": float(np.quantile(bootstrapped, 0.025)),
        "ci95_high": float(np.quantile(bootstrapped, 0.975)),
        "probability_improvement_positive": float(np.mean(bootstrapped > 0.0)),
    }


def _scores(
    prediction: np.ndarray,
    observed: np.ndarray,
    patch: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    power = max(float(np.mean(observed**2)), 1e-12)
    predicted_candidate = (patch["clean_answer_logits"][mask] + prediction).argmax(axis=1)
    direct_candidate = patch["intervened_answer_logits"][mask].argmax(axis=1)
    donor = patch["donor_id"][mask]
    return {
        "normalized_mse": float(np.mean((prediction - observed) ** 2) / power),
        "correlation": effect_correlation(prediction, observed),
        "answer_candidate_agreement": float(np.mean(predicted_candidate == direct_candidate)),
        "predicted_donor_candidate_rate": float(np.mean(predicted_candidate == donor)),
        "direct_donor_candidate_rate": float(np.mean(direct_candidate == donor)),
    }


__all__ = ["run_qwen_population_jacobian_confirmation"]
