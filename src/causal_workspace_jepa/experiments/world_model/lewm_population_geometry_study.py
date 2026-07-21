"""Population-versus-local causal transport in a recurrent pixel JEPA.

The experiment treats a categorical environment-action replacement as a finite
chord between two one-hot action vertices.  It compares the exact local
transition Jacobian with population and within-context derivative averages,
while retaining direct model execution as the target.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.func import jacrev, vmap

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.synthetic.pixel_tiny_maze import (
    render_pixel_tiny_maze,
    valid_positions,
)
from causal_workspace_jepa.data.synthetic.tiny_maze import step_tiny_maze
from causal_workspace_jepa.models.intervention_jepa import effect_correlation
from causal_workspace_jepa.models.lewm import SmallLeWorldModel


@dataclass(frozen=True)
class ContextBank:
    """Fixed initial states, goals, and suffixes for one rollout horizon."""

    images: np.ndarray
    positions: np.ndarray
    goals: np.ndarray
    goal_ids: np.ndarray
    suffix_ids: np.ndarray
    suffix_actions: np.ndarray


def run_lewm_population_geometry_study(config_path: str | Path) -> dict[str, Any]:
    """Execute the preregistered held-out-goal population-Jacobian study."""

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    hardware = require_free_disk(resource_profile)
    if not torch.cuda.is_available():
        raise RuntimeError("WM-POPULATION-JACOBIAN-001 requires CUDA")
    device = torch.device(str(config.get("device", "cuda")))
    seeds = [int(value) for value in config["model_seeds"]]
    analysis_seed = int(config.get("analysis_seed", 419))
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path.as_posix()}",
        resource_profile=resource_profile,
        seed=analysis_seed,
    )
    if provenance.git_dirty:
        raise RuntimeError("WM-POPULATION-JACOBIAN-001 requires a clean committed worktree")

    positions = valid_positions()
    train_goal_ids = tuple(int(value) for value in config["train_goal_ids"])
    validation_goal_ids = tuple(int(value) for value in config["validation_goal_ids"])
    test_goal_ids = tuple(int(value) for value in config["test_goal_ids"])
    _validate_goal_split(train_goal_ids, validation_goal_ids, test_goal_ids, len(positions))
    split_name = str(config.get("analysis_split", "validation"))
    if split_name != "validation":
        raise ValueError("v1 is frozen to the previously unanalyzed validation-goal split")
    evaluation_goal_ids = validation_goal_ids
    horizons = tuple(int(value) for value in config.get("horizons", [1, 4]))
    suffix_templates = np.asarray(config["horizon4_suffix_action_ids"], dtype=np.int64).reshape(-1, 3)
    checkpoints = _checkpoint_records(config, seeds)
    qwen_reference = json.loads(Path(str(config["qwen_confirmation_metrics"])).read_text("utf-8"))
    qwen_confirmed = bool(all(qwen_reference["hypothesis_decisions"].values()))

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = checkpoints[seed]
        model = SmallLeWorldModel.load(checkpoint["path"], map_location=device).to(device).eval()
        decoder = _fit_physical_decoder(
            model,
            tuple(positions[index] for index in train_goal_ids),
            cell_size=int(config.get("cell_size", 4)),
            ridge=float(config.get("decoder_ridge", 1e-3)),
            device=device,
        )
        horizon_results: dict[str, Any] = {}
        for horizon in horizons:
            suffixes = _suffixes_for_horizon(horizon, suffix_templates)
            train_bank = _make_context_bank(
                train_goal_ids,
                positions,
                suffixes,
                cell_size=int(config.get("cell_size", 4)),
            )
            evaluation_bank = _make_context_bank(
                evaluation_goal_ids,
                positions,
                suffixes,
                cell_size=int(config.get("cell_size", 4)),
            )
            horizon_results[str(horizon)] = _audit_horizon(
                config,
                model,
                decoder,
                train_bank,
                evaluation_bank,
                horizon,
                seed,
                device,
            )
        h4 = horizon_results["4"]
        h1 = horizon_results["1"]
        population_pass = bool(
            h4["scores"]["population"]["decoded_physics"]["normalized_mse"]
            <= h4["scores"]["local"]["decoded_physics"]["normalized_mse"]
            * float(config.get("population_mse_ratio_max", 0.80))
            and h4["scores"]["population"]["decoded_physics"]["correlation"]
            >= h4["scores"]["local"]["decoded_physics"]["correlation"]
            + float(config.get("population_correlation_margin_min", 0.02))
            and h4["per_goal"]["population_better_decoded_mse_goals"]
            >= int(config.get("population_better_goals_min", 4))
        )
        curve = h4["averaging_curve"]
        curve_sizes = sorted(int(value) for value in curve)
        curve_mse = np.asarray(
            [curve[str(size)]["median_decoded_normalized_mse"] for size in curve_sizes]
        )
        log_curve_correlation = float(np.corrcoef(np.log(curve_sizes), curve_mse)[0, 1])
        curve_pass = bool(
            curve["256"]["median_decoded_normalized_mse"]
            <= curve["1"]["median_decoded_normalized_mse"]
            * float(config.get("curve_mse_ratio_max", 0.85))
            and log_curve_correlation
            <= float(config.get("curve_log_correlation_max", -0.80))
        )
        vertex_pass = bool(
            all(
                horizon_results[str(h)]["scores"]["context_vertex_mean"]["decoded_physics"]
                ["normalized_mse"]
                <= horizon_results[str(h)]["scores"]["local"]["decoded_physics"]
                ["normalized_mse"]
                * float(config.get("vertex_mean_mse_ratio_max", 0.80))
                for h in horizons
            )
        )
        semantic_null_pass = bool(
            h4["scores"]["population"]["decoded_physics"]["normalized_mse"]
            <= h4["action_column_permutation_null"]["p05_decoded_normalized_mse"]
            * float(config.get("semantic_null_mse_ratio_max", 0.80))
        )
        numerical_pass = bool(
            all(
                horizon_results[str(h)]["path_integral_audit"]["max_relative_error"]
                <= float(config.get("path_integral_relative_error_max", 1e-4))
                and horizon_results[str(h)]["gauge_audit"]["decoded_invariance_max_abs"]
                <= float(config.get("gauge_invariance_max_abs", 1e-10))
                and horizon_results[str(h)]["gauge_audit"]["analytic_ranking_flip"]
                for h in horizons
            )
        )
        seed_results.append(
            {
                "model_seed": seed,
                "checkpoint": checkpoint,
                "decoder": decoder["metrics"],
                "horizons": horizon_results,
                "hypothesis_decisions": {
                    "h_wm_11_population_transition_smoothing": population_pass,
                    "h_wm_12_averaging_curve": curve_pass,
                    "h_wm_13_within_context_path_averaging": vertex_pass,
                    "h_wm_14_action_semantic_specificity": semantic_null_pass,
                    "h_geo_07_gauge_safe_fidelity": numerical_pass,
                },
                "horizon4_curve_log_size_mse_correlation": log_curve_correlation,
                "horizon1_population_to_local_decoded_mse_ratio": (
                    h1["scores"]["population"]["decoded_physics"]["normalized_mse"]
                    / max(
                        h1["scores"]["local"]["decoded_physics"]["normalized_mse"], 1e-12
                    )
                ),
                "horizon4_population_to_local_decoded_mse_ratio": (
                    h4["scores"]["population"]["decoded_physics"]["normalized_mse"]
                    / max(
                        h4["scores"]["local"]["decoded_physics"]["normalized_mse"], 1e-12
                    )
                ),
            }
        )

    required = int(config.get("required_passing_seeds", 2))
    decision_names = tuple(seed_results[0]["hypothesis_decisions"])
    decision_counts = {
        name: sum(bool(result["hypothesis_decisions"][name]) for result in seed_results)
        for name in decision_names
    }
    aggregate_decisions = {name: count >= required for name, count in decision_counts.items()}
    cross_domain = bool(
        qwen_confirmed
        and aggregate_decisions["h_wm_11_population_transition_smoothing"]
        and aggregate_decisions["h_wm_12_averaging_curve"]
    )
    aggregate_decisions["h_cross_02_finite_chord_population_transport"] = cross_domain
    scientific_passes = [
        aggregate_decisions["h_wm_11_population_transition_smoothing"],
        aggregate_decisions["h_wm_12_averaging_curve"],
        aggregate_decisions["h_wm_13_within_context_path_averaging"],
        aggregate_decisions["h_wm_14_action_semantic_specificity"],
    ]
    numerical_ok = aggregate_decisions["h_geo_07_gauge_safe_fidelity"]
    status = (
        "REJECTED_NUMERICAL_GATE"
        if not numerical_ok
        else "COMPLETED_POSITIVE"
        if all(scientific_passes)
        else "COMPLETED_MIXED"
        if any(scientific_passes)
        else "COMPLETED_NEGATIVE"
    )
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-POPULATION-JACOBIAN-001")),
        "status": status,
        "evidence_level": "Generalization",
        "model": "faithful small LeWorldModel reproduction",
        "model_seeds": seeds,
        "required_passing_seeds": required,
        "goal_split": {
            "seed": int(config.get("goal_split_seed", 401)),
            "train_goal_ids": list(train_goal_ids),
            "validation_goal_ids": list(validation_goal_ids),
            "reserved_test_goal_ids": list(test_goal_ids),
            "analysis_split": split_name,
            "goal_positions": [list(value) for value in positions],
        },
        "intervention": {
            "type": "valid categorical environment-action replacement",
            "source": "first action in one- or four-step autoregressive rollout",
            "recipient_donor_pairs_per_context": 12,
            "horizons": list(horizons),
            "horizon4_suffix_action_ids": suffix_templates.tolist(),
        },
        "qwen_confirmation_reference": {
            "path": str(config["qwen_confirmation_metrics"]),
            "experiment_id": qwen_reference["experiment_id"],
            "all_registered_decisions_pass": qwen_confirmed,
        },
        "decision_counts": decision_counts,
        "hypothesis_decisions": aggregate_decisions,
        "seed_results": seed_results,
        "hardware": hardware.as_dict(),
        "runtime_seconds": float(time.perf_counter() - started),
        "scientific_boundary": (
            "The target is the frozen JEPA's directly executed latent transition under valid "
            "one-hot action swaps. Decoded physics is a train-goal linear readout and is used to "
            "make the primary comparison invariant to an invertible latent coordinate change. "
            "Planner agreement is reported only when the direct frozen planner clears the fixed "
            "competence gate. This study does not establish a workspace, a semantic latent axis, "
            "or novelty of corpus-averaged Jacobians."
        ),
    }
    output = Path(
        str(config.get("output_metrics", "artifacts/metrics/lewm_population_geometry_v1.json"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": output.as_posix(), "hypothesis_decisions": aggregate_decisions},
    )
    return metrics


def _checkpoint_records(config: dict[str, Any], seeds: Iterable[int]) -> dict[int, dict[str, Any]]:
    directory = Path(str(config["checkpoint_dir"]))
    expected = {int(key): str(value).lower() for key, value in config["checkpoint_sha256"].items()}
    records: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = directory / f"seed_{seed}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[seed]:
            raise RuntimeError(f"checkpoint checksum mismatch for seed {seed}: {digest}")
        records[seed] = {"path": path.as_posix(), "sha256": digest, "bytes": path.stat().st_size}
    return records


def _validate_goal_split(
    train: tuple[int, ...], validation: tuple[int, ...], test: tuple[int, ...], count: int
) -> None:
    combined = train + validation + test
    if len(combined) != count or set(combined) != set(range(count)):
        raise ValueError("goal splits must partition every valid goal exactly once")
    if len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError("goal splits must not contain duplicates")


def _suffixes_for_horizon(horizon: int, horizon4: np.ndarray) -> np.ndarray:
    if horizon == 1:
        return np.empty((1, 0), dtype=np.int64)
    if horizon == 4 and horizon4.ndim == 2 and horizon4.shape[1] == 3:
        return horizon4
    raise ValueError("this preregistration permits only horizon 1 and fixed horizon-4 suffixes")


def _make_context_bank(
    goal_ids: tuple[int, ...],
    all_positions: tuple[tuple[int, int], ...],
    suffix_templates: np.ndarray,
    *,
    cell_size: int,
) -> ContextBank:
    images: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = []
    goal_rows: list[int] = []
    suffix_ids: list[int] = []
    suffix_actions: list[np.ndarray] = []
    eye = np.eye(4, dtype=np.float32)
    for goal_id in goal_ids:
        goal = all_positions[goal_id]
        for position in all_positions:
            for suffix_id, suffix in enumerate(suffix_templates):
                images.append(render_pixel_tiny_maze(position, goal, cell_size=cell_size))
                positions.append(position)
                goals.append(goal)
                goal_rows.append(goal_id)
                suffix_ids.append(suffix_id)
                suffix_actions.append(eye[suffix] if len(suffix) else np.empty((0, 4), np.float32))
    return ContextBank(
        images=np.asarray(images, dtype=np.float32),
        positions=np.asarray(positions, dtype=np.int64),
        goals=np.asarray(goals, dtype=np.int64),
        goal_ids=np.asarray(goal_rows, dtype=np.int64),
        suffix_ids=np.asarray(suffix_ids, dtype=np.int64),
        suffix_actions=np.asarray(suffix_actions, dtype=np.float32),
    )


def _encode(model: SmallLeWorldModel, images: np.ndarray, device: torch.device) -> Tensor:
    values: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), 512):
            encoded, _ = model.encode_pixels(torch.as_tensor(images[start : start + 512], device=device))
            values.append(encoded)
    return torch.cat(values)


def _fit_physical_decoder(
    model: SmallLeWorldModel,
    goals: tuple[tuple[int, int], ...],
    *,
    cell_size: int,
    ridge: float,
    device: torch.device,
) -> dict[str, Any]:
    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    positions = valid_positions()
    for goal in goals:
        goal_array = np.asarray(goal)
        for position in positions:
            for action in range(4):
                following = step_tiny_maze(np.asarray(position), action)
                images.append(render_pixel_tiny_maze(following, goal, cell_size=cell_size))
                targets.append(
                    np.asarray(
                        [following[0], following[1], np.abs(following - goal_array).sum()],
                        dtype=np.float64,
                    )
                )
    features = _encode(model, np.asarray(images, np.float32), device).cpu().numpy().astype(np.float64)
    target = np.asarray(targets, dtype=np.float64)
    design = np.concatenate((features, np.ones((len(features), 1))), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    prediction = design @ weights
    baseline = np.mean((target - target.mean(axis=0, keepdims=True)) ** 2)
    r2 = 1.0 - float(np.mean((prediction - target) ** 2)) / max(float(baseline), 1e-12)
    return {
        "weights": weights[:-1],
        "bias": weights[-1],
        "metrics": {
            "fit_examples": len(features),
            "outputs": ["next_x", "next_y", "next_manhattan_distance"],
            "train_r2": r2,
            "ridge": ridge,
            "evidence_level": "Availability",
        },
    }


def _transition_outputs(
    model: SmallLeWorldModel,
    initial: Tensor,
    suffix: Tensor,
    *,
    batch_size: int = 512,
) -> Tensor:
    eye = torch.eye(4, device=initial.device, dtype=initial.dtype)
    expanded_initial = initial.repeat_interleave(4, dim=0)
    first = eye.repeat(len(initial), 1)
    expanded_suffix = suffix.repeat_interleave(4, dim=0)
    actions = torch.cat((first[:, None], expanded_suffix), dim=1)
    outputs: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(actions), batch_size):
            trajectory, _ = model.rollout(
                expanded_initial[start : start + batch_size], actions[start : start + batch_size]
            )
            outputs.append(trajectory[:, -1])
    return torch.cat(outputs).reshape(len(initial), 4, -1)


def _transition_jacobians(
    model: SmallLeWorldModel,
    initial: Tensor,
    suffix: Tensor,
    actions: Tensor,
    *,
    chunk_size: int,
) -> Tensor:
    def transition(z: Tensor, first: Tensor, later: Tensor) -> Tensor:
        full_actions = torch.cat((first[None], later), dim=0)
        trajectory, _ = model.rollout(z[None], full_actions[None])
        return trajectory[0, -1]

    return vmap(jacrev(transition, argnums=1), chunk_size=chunk_size)(initial, actions, suffix)


def _audit_horizon(
    config: dict[str, Any],
    model: SmallLeWorldModel,
    decoder: dict[str, Any],
    train_bank: ContextBank,
    evaluation_bank: ContextBank,
    horizon: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    eye = torch.eye(4, device=device)
    train_initial = _encode(model, train_bank.images, device)
    evaluation_initial = _encode(model, evaluation_bank.images, device)
    train_suffix = torch.as_tensor(train_bank.suffix_actions, device=device)
    evaluation_suffix = torch.as_tensor(evaluation_bank.suffix_actions, device=device)
    train_anchor_initial = train_initial.repeat_interleave(4, dim=0)
    evaluation_anchor_initial = evaluation_initial.repeat_interleave(4, dim=0)
    train_anchor_suffix = train_suffix.repeat_interleave(4, dim=0)
    evaluation_anchor_suffix = evaluation_suffix.repeat_interleave(4, dim=0)
    anchor_actions = eye.repeat(len(train_bank.images), 1)
    evaluation_actions = eye.repeat(len(evaluation_bank.images), 1)
    chunk_size = int(config.get("jacobian_chunk_size", 16))
    train_jacobians = _transition_jacobians(
        model,
        train_anchor_initial,
        train_anchor_suffix,
        anchor_actions,
        chunk_size=chunk_size,
    ).reshape(len(train_bank.images), 4, -1, 4)
    evaluation_jacobians = _transition_jacobians(
        model,
        evaluation_anchor_initial,
        evaluation_anchor_suffix,
        evaluation_actions,
        chunk_size=chunk_size,
    ).reshape(len(evaluation_bank.images), 4, -1, 4)
    centroid_actions = torch.full((len(evaluation_bank.images), 4), 0.25, device=device)
    centroid_jacobians = _transition_jacobians(
        model,
        evaluation_initial,
        evaluation_suffix,
        centroid_actions,
        chunk_size=chunk_size,
    )
    outputs = _transition_outputs(model, evaluation_initial, evaluation_suffix)
    goal_images = np.asarray(
        [
            render_pixel_tiny_maze(goal, goal, cell_size=int(config.get("cell_size", 4)))
            for goal in evaluation_bank.goals
        ],
        dtype=np.float32,
    )
    goal_embeddings = _encode(model, goal_images, device)

    pairs = _action_pairs()
    recipients = torch.as_tensor([pair[0] for pair in pairs], device=device)
    donors = torch.as_tensor([pair[1] for pair in pairs], device=device)
    delta = eye[donors] - eye[recipients]
    observed = outputs[:, donors] - outputs[:, recipients]
    local = torch.einsum("npda,pa->npd", evaluation_jacobians[:, recipients], delta)
    population_jacobian = train_jacobians.mean(dim=(0, 1))
    population = torch.einsum("da,pa->pd", population_jacobian, delta)[None].expand_as(observed)
    vertex_jacobian = evaluation_jacobians.mean(dim=1)
    vertex = torch.einsum("nda,pa->npd", vertex_jacobian, delta)
    centroid = torch.einsum("nda,pa->npd", centroid_jacobians, delta)
    trapezoid = 0.5 * torch.einsum(
        "npda,pa->npd",
        evaluation_jacobians[:, recipients] + evaluation_jacobians[:, donors],
        delta,
    )
    observed_np = observed.detach().cpu().numpy().reshape(-1, observed.shape[-1]).astype(np.float64)
    predictions = {
        "local": local,
        "population": population,
        "context_vertex_mean": vertex,
        "context_centroid": centroid,
        "endpoint_trapezoid": trapezoid,
    }
    decoder_weights = np.asarray(decoder["weights"], dtype=np.float64)
    scores = {
        name: _multi_endpoint_scores(
            value.detach().cpu().numpy().reshape(-1, value.shape[-1]).astype(np.float64),
            observed_np,
            decoder_weights,
        )
        for name, value in predictions.items()
    }
    goal_rows = np.repeat(evaluation_bank.goal_ids, len(pairs))
    per_goal = _per_goal_scores(
        predictions["local"].detach().cpu().numpy().reshape(-1, observed.shape[-1]),
        predictions["population"].detach().cpu().numpy().reshape(-1, observed.shape[-1]),
        observed_np,
        decoder_weights,
        goal_rows,
    )
    planner = _planner_audit(
        evaluation_bank,
        outputs.detach().cpu().numpy().astype(np.float64),
        evaluation_jacobians.detach().cpu().numpy().astype(np.float64),
        population_jacobian.detach().cpu().numpy().astype(np.float64),
        goal_embeddings.detach().cpu().numpy().astype(np.float64),
        competence_min=float(config.get("planner_competence_min", 0.60)),
    )
    averaging_curve = _averaging_curve(
        config,
        train_jacobians.detach().cpu().numpy().reshape(-1, observed.shape[-1], 4),
        np.asarray([np.eye(4)[d] - np.eye(4)[r] for r, d in pairs], dtype=np.float64),
        observed_np,
        decoder_weights,
        seed=seed + 1000 * horizon,
    )
    column_null = _action_column_permutation_null(
        population_jacobian.detach().cpu().numpy().astype(np.float64),
        np.asarray([np.eye(4)[d] - np.eye(4)[r] for r, d in pairs], dtype=np.float64),
        observed_np,
        decoder_weights,
        contexts=len(evaluation_bank.images),
    )
    path_audit = _path_integral_audit(
        config,
        model,
        evaluation_initial,
        evaluation_suffix,
        outputs,
        pairs,
        horizon=horizon,
        seed=seed,
        chunk_size=chunk_size,
    )
    gauge = _gauge_audit(
        predictions["local"].detach().cpu().numpy().reshape(-1, observed.shape[-1]),
        predictions["population"].detach().cpu().numpy().reshape(-1, observed.shape[-1]),
        observed_np,
        decoder_weights,
        seed=seed + 3000 * horizon,
    )
    return {
        "horizon": horizon,
        "train_contexts": len(train_bank.images),
        "evaluation_contexts": len(evaluation_bank.images),
        "finite_action_swaps": int(observed_np.shape[0]),
        "scores": scores,
        "per_goal": per_goal,
        "planner": planner,
        "averaging_curve": averaging_curve,
        "action_column_permutation_null": column_null,
        "path_integral_audit": path_audit,
        "gauge_audit": gauge,
        "evidence_level": "Generalization",
    }


def _action_pairs() -> tuple[tuple[int, int], ...]:
    return tuple((recipient, donor) for recipient in range(4) for donor in range(4) if donor != recipient)


def _multi_endpoint_scores(
    prediction: np.ndarray, observed: np.ndarray, decoder: np.ndarray
) -> dict[str, dict[str, float]]:
    return {
        "latent": _transport_scores(prediction, observed),
        "decoded_physics": _transport_scores(prediction @ decoder, observed @ decoder),
    }


def _transport_scores(prediction: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    power = max(float(np.mean(observed**2)), 1e-12)
    numerator = np.sum(prediction * observed, axis=1)
    cosine = numerator / np.maximum(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(observed, axis=1), 1e-12
    )
    return {
        "normalized_mse": float(np.mean((prediction - observed) ** 2) / power),
        "correlation": effect_correlation(prediction, observed),
        "mean_cosine": float(np.mean(cosine)),
        "recovery": float(1.0 - np.mean((prediction - observed) ** 2) / power),
    }


def _per_goal_scores(
    local: np.ndarray,
    population: np.ndarray,
    observed: np.ndarray,
    decoder: np.ndarray,
    goal_ids: np.ndarray,
) -> dict[str, Any]:
    rows = []
    better = 0
    for goal_id in sorted(set(int(value) for value in goal_ids)):
        mask = goal_ids == goal_id
        local_score = _transport_scores(local[mask] @ decoder, observed[mask] @ decoder)
        population_score = _transport_scores(population[mask] @ decoder, observed[mask] @ decoder)
        wins = population_score["normalized_mse"] < local_score["normalized_mse"]
        better += int(wins)
        rows.append(
            {
                "goal_id": goal_id,
                "local_decoded_normalized_mse": local_score["normalized_mse"],
                "population_decoded_normalized_mse": population_score["normalized_mse"],
                "population_better": wins,
            }
        )
    return {"population_better_decoded_mse_goals": better, "goals": rows}


def _averaging_curve(
    config: dict[str, Any],
    train_jacobians: np.ndarray,
    pair_deltas: np.ndarray,
    observed: np.ndarray,
    decoder: np.ndarray,
    *,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    contexts = observed.shape[0] // len(pair_deltas)
    sizes = [1, 4, 16, 64, 256, len(train_jacobians)]
    draws = int(config.get("averaging_curve_draws", 64))
    result: dict[str, dict[str, float | int]] = {}
    for size in sizes:
        actual_draws = 1 if size == len(train_jacobians) else draws
        values = []
        for _ in range(actual_draws):
            indices = rng.choice(len(train_jacobians), size=size, replace=False)
            mean_jacobian = train_jacobians[indices].mean(axis=0)
            one_context = np.einsum("da,pa->pd", mean_jacobian, pair_deltas)
            prediction = np.tile(one_context, (contexts, 1))
            values.append(_transport_scores(prediction @ decoder, observed @ decoder)["normalized_mse"])
        array = np.asarray(values)
        result[str(size)] = {
            "draws": actual_draws,
            "median_decoded_normalized_mse": float(np.median(array)),
            "p05_decoded_normalized_mse": float(np.quantile(array, 0.05)),
            "p95_decoded_normalized_mse": float(np.quantile(array, 0.95)),
        }
    return result


def _action_column_permutation_null(
    population_jacobian: np.ndarray,
    pair_deltas: np.ndarray,
    observed: np.ndarray,
    decoder: np.ndarray,
    *,
    contexts: int,
) -> dict[str, float | int]:
    values = []
    for permutation in itertools.permutations(range(4)):
        if permutation == (0, 1, 2, 3):
            continue
        permuted = population_jacobian[:, permutation]
        one_context = np.einsum("da,pa->pd", permuted, pair_deltas)
        prediction = np.tile(one_context, (contexts, 1))
        values.append(_transport_scores(prediction @ decoder, observed @ decoder)["normalized_mse"])
    array = np.asarray(values)
    return {
        "nonidentity_permutations": len(values),
        "mean_decoded_normalized_mse": float(array.mean()),
        "p05_decoded_normalized_mse": float(np.quantile(array, 0.05)),
        "p95_decoded_normalized_mse": float(np.quantile(array, 0.95)),
    }


def _path_integral_audit(
    config: dict[str, Any],
    model: SmallLeWorldModel,
    initial: Tensor,
    suffix: Tensor,
    outputs: Tensor,
    pairs: tuple[tuple[int, int], ...],
    *,
    horizon: int,
    seed: int,
    chunk_size: int,
) -> dict[str, float | int]:
    count = min(int(config.get("path_audit_pairs", 128)), len(initial) * len(pairs))
    rng = np.random.default_rng(int(config.get("path_audit_seed", 431)) + seed + horizon)
    selected = rng.choice(len(initial) * len(pairs), size=count, replace=False)
    context_ids = selected // len(pairs)
    pair_ids = selected % len(pairs)
    recipient = np.asarray([pairs[index][0] for index in pair_ids], dtype=np.int64)
    donor = np.asarray([pairs[index][1] for index in pair_ids], dtype=np.int64)
    nodes, weights = np.polynomial.legendre.leggauss(int(config.get("path_quadrature_nodes", 12)))
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    eye = torch.eye(4, device=initial.device)
    delta = eye[torch.as_tensor(donor, device=initial.device)] - eye[
        torch.as_tensor(recipient, device=initial.device)
    ]
    recipient_action = eye[torch.as_tensor(recipient, device=initial.device)]
    actions = recipient_action[:, None, :] + torch.as_tensor(
        nodes, device=initial.device, dtype=initial.dtype
    )[None, :, None] * delta[:, None, :]
    repeated_initial = initial[torch.as_tensor(context_ids, device=initial.device)].repeat_interleave(
        len(nodes), dim=0
    )
    repeated_suffix = suffix[torch.as_tensor(context_ids, device=initial.device)].repeat_interleave(
        len(nodes), dim=0
    )
    jacobians = _transition_jacobians(
        model,
        repeated_initial,
        repeated_suffix,
        actions.reshape(-1, 4),
        chunk_size=chunk_size,
    ).reshape(count, len(nodes), -1, 4)
    directional = torch.einsum("ntda,na->ntd", jacobians, delta)
    integrated = torch.einsum(
        "t,ntd->nd", torch.as_tensor(weights, device=initial.device, dtype=initial.dtype), directional
    )
    direct = outputs[
        torch.as_tensor(context_ids, device=initial.device),
        torch.as_tensor(donor, device=initial.device),
    ] - outputs[
        torch.as_tensor(context_ids, device=initial.device),
        torch.as_tensor(recipient, device=initial.device),
    ]
    relative = (integrated - direct).norm(dim=1) / direct.norm(dim=1).clamp_min(1e-8)
    return {
        "pairs": count,
        "quadrature_nodes": len(nodes),
        "median_relative_error": float(relative.median().detach().cpu()),
        "p95_relative_error": float(torch.quantile(relative, 0.95).detach().cpu()),
        "max_relative_error": float(relative.max().detach().cpu()),
    }


def _planner_audit(
    bank: ContextBank,
    outputs: np.ndarray,
    local_jacobians: np.ndarray,
    population_jacobian: np.ndarray,
    goal_embeddings: np.ndarray,
    *,
    competence_min: float,
) -> dict[str, float | bool | str]:
    eye = np.eye(4, dtype=np.float64)
    direct_cost = np.mean((outputs - goal_embeddings[:, None, :]) ** 2, axis=2)
    direct_action = direct_cost.argmin(axis=1)
    local_candidates = outputs[:, :1] + np.einsum(
        "nda,ca->ncd", local_jacobians[:, 0], eye - eye[0]
    )
    population_candidates = outputs[:, :1] + np.einsum(
        "da,ca->cd", population_jacobian, eye - eye[0]
    )[None]
    local_action = np.mean((local_candidates - goal_embeddings[:, None, :]) ** 2, axis=2).argmin(1)
    population_action = np.mean(
        (population_candidates - goal_embeddings[:, None, :]) ** 2, axis=2
    ).argmin(1)
    true_costs = np.zeros((len(bank.positions), 4), dtype=np.float64)
    suffix_ids = bank.suffix_actions.argmax(axis=2) if bank.suffix_actions.shape[1] else None
    for row, (position, goal) in enumerate(zip(bank.positions, bank.goals, strict=True)):
        for first in range(4):
            current = step_tiny_maze(position, first)
            if suffix_ids is not None:
                for action in suffix_ids[row]:
                    current = step_tiny_maze(current, int(action))
            true_costs[row, first] = np.abs(current - goal).sum()
    optimal = true_costs == true_costs.min(axis=1, keepdims=True)
    direct_competence = float(np.mean(optimal[np.arange(len(optimal)), direct_action]))
    eligible = direct_competence >= competence_min
    return {
        "direct_environment_optimal_rate": direct_competence,
        "competence_threshold": competence_min,
        "behavior_claim_eligible": eligible,
        "local_direct_action_agreement": float(np.mean(local_action == direct_action)),
        "population_direct_action_agreement": float(np.mean(population_action == direct_action)),
        "local_environment_optimal_rate": float(np.mean(optimal[np.arange(len(optimal)), local_action])),
        "population_environment_optimal_rate": float(
            np.mean(optimal[np.arange(len(optimal)), population_action])
        ),
        "evidence_level": "Specificity" if eligible else "Availability",
        "boundary": (
            "action-selection comparisons are evidential only if behavior_claim_eligible is true"
        ),
    }


def _gauge_audit(
    local: np.ndarray,
    population: np.ndarray,
    observed: np.ndarray,
    decoder: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    diagonal = np.geomspace(0.1, 10.0, observed.shape[1])
    rng.shuffle(diagonal)
    transform = np.diag(diagonal)
    inverse = np.diag(1.0 / diagonal)
    transformed_decoder = inverse @ decoder
    transformed_observed = observed @ transform.T
    transformed_local = local @ transform.T
    transformed_population = population @ transform.T
    original_decoded = np.concatenate((local @ decoder, population @ decoder), axis=0)
    transformed_decoded = np.concatenate(
        (
            transformed_local @ transformed_decoder,
            transformed_population @ transformed_decoder,
        ),
        axis=0,
    )
    original_naive = {
        "local": _transport_scores(local, observed)["normalized_mse"],
        "population": _transport_scores(population, observed)["normalized_mse"],
    }
    transformed_naive = {
        "local": _transport_scores(transformed_local, transformed_observed)["normalized_mse"],
        "population": _transport_scores(transformed_population, transformed_observed)[
            "normalized_mse"
        ],
    }
    analytic_observed = np.asarray([[1.0, 1.0]])
    analytic_local = np.asarray([[1.0, 1.5]])
    analytic_population = np.asarray([[1.2, 1.0]])
    analytic_transform = np.diag([10.0, 1.0])
    before = (
        _transport_scores(analytic_population, analytic_observed)["normalized_mse"]
        < _transport_scores(analytic_local, analytic_observed)["normalized_mse"]
    )
    after = (
        _transport_scores(
            analytic_population @ analytic_transform.T,
            analytic_observed @ analytic_transform.T,
        )["normalized_mse"]
        > _transport_scores(
            analytic_local @ analytic_transform.T,
            analytic_observed @ analytic_transform.T,
        )["normalized_mse"]
    )
    return {
        "diagonal_condition_number": float(diagonal.max() / diagonal.min()),
        "decoded_invariance_max_abs": float(np.max(np.abs(original_decoded - transformed_decoded))),
        "naive_local_normalized_mse_before": original_naive["local"],
        "naive_local_normalized_mse_after": transformed_naive["local"],
        "naive_population_normalized_mse_before": original_naive["population"],
        "naive_population_normalized_mse_after": transformed_naive["population"],
        "analytic_ranking_flip": bool(before and after),
    }


__all__ = [
    "run_lewm_population_geometry_study",
    "_action_pairs",
    "_gauge_audit",
    "_transport_scores",
]
