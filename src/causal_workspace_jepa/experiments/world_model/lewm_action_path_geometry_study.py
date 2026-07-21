"""Pathwise action-Jacobian geometry in a recurrent pixel JEPA.

The calibration mode is restricted to previously exposed validation goals.  It
profiles exact directional derivatives along valid one-hot action chords in a
train-goal decoded physical coordinate system.  A separately preregistered
test mode may consume the protected test goals after thresholds are frozen.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.synthetic.pixel_tiny_maze import valid_positions
from causal_workspace_jepa.experiments.world_model.lewm_population_geometry_study import (
    _action_pairs,
    _checkpoint_records,
    _encode,
    _fit_physical_decoder,
    _make_context_bank,
    _suffixes_for_horizon,
    _transition_jacobians,
    _transition_outputs,
    _validate_goal_split,
)
from causal_workspace_jepa.models.lewm import SmallLeWorldModel


def run_lewm_action_path_geometry_study(config_path: str | Path) -> dict[str, Any]:
    """Run validation calibration or a preregistered protected-test study."""

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    hardware = require_free_disk(
        str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    )
    if not torch.cuda.is_available():
        raise RuntimeError("LeWorldModel action-path geometry requires CUDA")
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path.as_posix()}",
        resource_profile=str(config["resource_profile"]),
        seed=int(config["analysis_seed"]),
    )
    if provenance.git_dirty:
        raise RuntimeError("LeWorldModel action-path geometry requires a clean worktree")

    experiment_id = str(config["id"])
    output = Path(str(config["output_metrics"]))
    progress_path = Path(
        str(config.get("progress_metrics", output.with_suffix(".progress.json")))
    )
    if progress_path == output:
        raise ValueError("progress_metrics must differ from output_metrics")
    run_fingerprint = _run_fingerprint(config_path, provenance.git_commit)
    progress_seed_results = _load_progress(
        progress_path,
        experiment_id=experiment_id,
        run_fingerprint=run_fingerprint,
    )
    progress_by_seed = {
        int(result["model_seed"]): result for result in progress_seed_results
    }
    if len(progress_by_seed) != len(progress_seed_results):
        raise RuntimeError(f"duplicate model seeds in progress: {progress_path}")
    loaded_seed_horizons = sum(
        len(result.get("horizons", {})) for result in progress_seed_results
    )
    split_name = str(config["analysis_split"])
    if experiment_id in {
        "WM-ACTION-PATH-CALIBRATION-001",
        "WM-ACTION-PATH-CALIBRATION-002",
    } and split_name != "validation":
        raise ValueError("calibration is restricted to previously exposed validation goals")
    if experiment_id == "WM-ACTION-PATH-GEOMETRY-001" and split_name != "test":
        raise ValueError("the preregistered study is restricted to protected test goals")
    positions = valid_positions()
    train_goal_ids = tuple(int(value) for value in config["train_goal_ids"])
    validation_goal_ids = tuple(int(value) for value in config["validation_goal_ids"])
    test_goal_ids = tuple(int(value) for value in config["test_goal_ids"])
    _validate_goal_split(train_goal_ids, validation_goal_ids, test_goal_ids, len(positions))
    evaluation_goal_ids = validation_goal_ids if split_name == "validation" else test_goal_ids
    seeds = tuple(int(value) for value in config["model_seeds"])
    unexpected_progress_seeds = set(progress_by_seed) - set(seeds)
    if unexpected_progress_seeds:
        raise RuntimeError(
            f"unexpected model seeds in progress: {sorted(unexpected_progress_seeds)}"
        )
    checkpoints = _checkpoint_records(config, seeds)
    suffix_templates = np.asarray(
        config["horizon4_suffix_action_ids"], dtype=np.int64
    ).reshape(-1, 3)
    device = torch.device(str(config.get("device", "cuda")))
    torch.manual_seed(int(config["analysis_seed"]))
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    seed_results = []
    for seed in seeds:
        resumed = progress_by_seed.get(seed)
        if resumed is not None and set(resumed.get("horizons", {})) == {"1", "4"}:
            seed_results.append(resumed)
            continue
        checkpoint = checkpoints[seed]
        model = SmallLeWorldModel.load(checkpoint["path"], map_location=device).to(device).eval()
        decoder = _fit_physical_decoder(
            model,
            tuple(positions[index] for index in train_goal_ids),
            cell_size=int(config.get("cell_size", 4)),
            ridge=float(config.get("decoder_ridge", 1e-3)),
            device=device,
        )
        horizon_results = dict(resumed.get("horizons", {})) if resumed is not None else {}
        for horizon in (1, 4):
            if str(horizon) in horizon_results:
                continue
            bank = _make_context_bank(
                evaluation_goal_ids,
                positions,
                _suffixes_for_horizon(horizon, suffix_templates),
                cell_size=int(config.get("cell_size", 4)),
            )
            horizon_results[str(horizon)] = _profile_horizon(
                config,
                model,
                decoder,
                bank,
                horizon=horizon,
                seed=seed,
                device=device,
            )
            partial = {
                "model_seed": seed,
                "checkpoint": checkpoint,
                "decoder": decoder["metrics"],
                "horizons": horizon_results,
            }
            progress_by_seed[seed] = partial
            _write_progress(
                progress_path,
                experiment_id=experiment_id,
                run_fingerprint=run_fingerprint,
                git_commit=provenance.git_commit,
                seed_results=[progress_by_seed[value] for value in seeds if value in progress_by_seed],
            )
        amplification = (
            horizon_results["4"]["summary"]["median_cancellation_ratio"]
            / max(
                horizon_results["1"]["summary"]["median_cancellation_ratio"],
                1e-12,
            )
        )
        completed_seed = {
            "model_seed": seed,
            "checkpoint": checkpoint,
            "decoder": decoder["metrics"],
            "horizons": horizon_results,
            "horizon4_to_horizon1_median_cancellation_ratio": amplification,
        }
        progress_by_seed[seed] = completed_seed
        seed_results.append(completed_seed)
        _write_progress(
            progress_path,
            experiment_id=experiment_id,
            run_fingerprint=run_fingerprint,
            git_commit=provenance.git_commit,
            seed_results=[progress_by_seed[value] for value in seeds if value in progress_by_seed],
        )

    decisions = _decide(config, seed_results) if split_name == "test" else {}
    metrics: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": (
            "CALIBRATION_ONLY"
            if split_name == "validation"
            else "COMPLETED_POSITIVE"
            if decisions and all(decisions.values())
            else "COMPLETED_MIXED"
            if any(decisions.values())
            else "COMPLETED_NEGATIVE"
        ),
        "evidence_level": "Availability" if split_name == "validation" else "Generalization",
        "model": "faithful small LeWorldModel reproduction",
        "model_seeds": list(seeds),
        "analysis_split": split_name,
        "goal_ids": list(evaluation_goal_ids),
        "protected_test_goals_touched": split_name == "test",
        "calibration_parent_metrics": config.get("calibration_parent_metrics"),
        "progress_resume": {
            "used": bool(progress_seed_results),
            "loaded_seed_horizons": loaded_seed_horizons,
        },
        "profile": {
            "contexts_per_action_pair": int(config["contexts_per_action_pair"]),
            "action_pairs": len(_action_pairs()),
            "gauss_order": int(config["gauss_order"]),
            "panel_refinements": list(config["panel_refinements"]),
            "endpoint": "train-goal decoded next_x/next_y/Manhattan effect",
        },
        "seed_results": seed_results,
        "hypothesis_decisions": decisions,
        "hardware": hardware.as_dict(),
        "runtime_seconds": float(time.perf_counter() - started),
        "scientific_boundary": (
            "Validation mode calibrates numerical and prospective thresholds only. Path length "
            "and cancellation are measured in a fixed train-goal decoded physical coordinate "
            "system, not raw latent Euclidean geometry. The study does not identify a circuit, "
            "workspace, semantic axis, or conscious process."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", "utf-8")
    write_provenance(
        output.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": output.as_posix(),
            "hypothesis_decisions": decisions,
            "progress_resume": metrics["progress_resume"],
        },
    )
    progress_path.unlink(missing_ok=True)
    return metrics


def _run_fingerprint(config_path: Path, git_commit: str) -> str:
    """Bind resumable progress to the exact configuration bytes and source commit."""

    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    digest.update(b"\0")
    digest.update(git_commit.encode("utf-8"))
    return digest.hexdigest()


def _load_progress(
    path: Path,
    *,
    experiment_id: str,
    run_fingerprint: str,
) -> list[dict[str, Any]]:
    """Load compatible progress or fail closed instead of mixing scientific runs."""

    if not path.exists():
        return []
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("experiment_id") != experiment_id:
        raise RuntimeError(f"progress experiment mismatch: {path}")
    if payload.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(f"stale progress fingerprint: {path}")
    results = payload.get("seed_results")
    if not isinstance(results, list):
        raise RuntimeError(f"invalid progress seed_results: {path}")
    return results


def _write_progress(
    path: Path,
    *,
    experiment_id: str,
    run_fingerprint: str,
    git_commit: str,
    seed_results: list[dict[str, Any]],
) -> None:
    """Atomically checkpoint completed horizons for safe same-run resumption."""

    payload = {
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "run_fingerprint": run_fingerprint,
        "seed_results": seed_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    temporary.replace(path)


def _profile_horizon(
    config: dict[str, Any],
    model: SmallLeWorldModel,
    decoder: dict[str, Any],
    bank: Any,
    *,
    horizon: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    initial = _encode(model, bank.images, device)
    suffix = torch.as_tensor(bank.suffix_actions, device=device)
    outputs = _transition_outputs(model, initial, suffix)
    pairs = _action_pairs()
    selected_contexts, pair_ids = _stratified_profile_indices(
        len(initial),
        contexts_per_pair=int(config["contexts_per_action_pair"]),
        seed=int(config["profile_seed"]) + seed + 1000 * horizon,
    )
    recipients = np.asarray([pairs[index][0] for index in pair_ids], dtype=np.int64)
    donors = np.asarray([pairs[index][1] for index in pair_ids], dtype=np.int64)
    eye = torch.eye(4, device=device, dtype=initial.dtype)
    context_tensor = torch.as_tensor(selected_contexts, device=device)
    recipient_tensor = torch.as_tensor(recipients, device=device)
    donor_tensor = torch.as_tensor(donors, device=device)
    delta = eye[donor_tensor] - eye[recipient_tensor]
    direct_latent = outputs[context_tensor, donor_tensor] - outputs[context_tensor, recipient_tensor]
    decoder_tensor = torch.as_tensor(decoder["weights"], device=device, dtype=initial.dtype)
    direct = (direct_latent @ decoder_tensor).detach().cpu()
    direct_norm = direct.norm(dim=1).clamp_min(float(config.get("direct_norm_floor", 1e-4)))

    refinements = tuple(int(value) for value in config["panel_refinements"])
    estimates = []
    final_directional = None
    final_weights = None
    for panels in refinements:
        nodes, weights = _composite_legendre(int(config["gauss_order"]), panels)
        node_tensor = torch.as_tensor(nodes, device=device, dtype=initial.dtype)
        actions = eye[recipient_tensor, None, :] + node_tensor[None, :, None] * delta[:, None, :]
        repeated_initial = initial[context_tensor].repeat_interleave(len(nodes), dim=0)
        repeated_suffix = suffix[context_tensor].repeat_interleave(len(nodes), dim=0)
        repeated_delta = delta[:, None, :].expand(-1, len(nodes), -1).reshape(-1, 4)
        directional = _decoded_directional_derivatives(
            model,
            repeated_initial,
            repeated_suffix,
            actions.reshape(-1, 4),
            repeated_delta,
            decoder_tensor,
            chunk_size=int(config.get("jacobian_chunk_size", 16)),
            outer_batch_size=int(config.get("jacobian_outer_batch_size", 64)),
        ).reshape(len(selected_contexts), len(nodes), -1)
        weight_tensor = torch.as_tensor(weights, dtype=directional.dtype)
        estimates.append(torch.einsum("t,ntd->nd", weight_tensor, directional))
        final_directional = directional
        final_weights = weight_tensor
        del actions, repeated_delta, repeated_initial, repeated_suffix
        torch.cuda.empty_cache()

    local = _decoded_directional_derivatives(
        model,
        initial[context_tensor],
        suffix[context_tensor],
        eye[recipient_tensor],
        delta,
        decoder_tensor,
        chunk_size=int(config.get("jacobian_chunk_size", 16)),
        outer_batch_size=int(config.get("jacobian_outer_batch_size", 64)),
    )
    final = estimates[-1]
    previous = estimates[-2]
    integration_error = (final - direct).norm(dim=1) / direct_norm
    refinement_change = (final - previous).norm(dim=1) / direct_norm
    local_error = (local - direct).norm(dim=1) / direct_norm
    assert final_directional is not None and final_weights is not None
    speed = final_directional.norm(dim=2)
    path_length = torch.einsum("t,nt->n", final_weights, speed)
    cancellation = path_length / direct_norm
    mean_speed = path_length
    peak_to_mean = speed.max(dim=1).values / mean_speed.clamp_min(1e-8)
    second_moment = torch.einsum("t,nt->n", final_weights, speed.square())
    effective_support = mean_speed.square() / second_moment.clamp_min(1e-8)

    arrays = {
        "direct_norm": _numpy(direct_norm),
        "cancellation_ratio": _numpy(cancellation),
        "local_relative_error": _numpy(local_error),
        "integration_relative_error": _numpy(integration_error),
        "refinement_relative_change": _numpy(refinement_change),
        "peak_to_mean_speed": _numpy(peak_to_mean),
        "effective_path_support": _numpy(effective_support),
    }
    correlation = _spearman(
        arrays["cancellation_ratio"], arrays["local_relative_error"]
    )
    null = _stratified_permutation_null(
        arrays["cancellation_ratio"],
        arrays["local_relative_error"],
        pair_ids,
        permutations=int(config.get("permutations", 256)),
        seed=int(config["permutation_seed"]) + seed + 1000 * horizon,
    )
    rows = []
    for index, (context_id, pair_id) in enumerate(
        zip(selected_contexts, pair_ids, strict=True)
    ):
        rows.append(
            {
                "context_id": int(context_id),
                "goal_id": int(bank.goal_ids[context_id]),
                "suffix_id": int(bank.suffix_ids[context_id]),
                "recipient_action": int(recipients[index]),
                "donor_action": int(donors[index]),
                **{name: float(values[index]) for name, values in arrays.items()},
            }
        )
    return {
        "horizon": horizon,
        "contexts": len(bank.images),
        "profiled_chords": len(rows),
        "summary": {
            **{
                f"{stat}_{name}": float(function(values))
                for name, values in arrays.items()
                for stat, function in (
                    ("median", np.median),
                    ("p95", lambda value: np.quantile(value, 0.95)),
                    ("max", np.max),
                )
            },
            "cancellation_local_error_spearman": correlation,
            "stratified_permutation_null": null,
        },
        "chords": rows,
    }


def _decoded_directional_derivatives(
    model: SmallLeWorldModel,
    initial: torch.Tensor,
    suffix: torch.Tensor,
    actions: torch.Tensor,
    delta: torch.Tensor,
    decoder: torch.Tensor,
    *,
    chunk_size: int,
    outer_batch_size: int,
) -> torch.Tensor:
    """Stream exact Jacobians, immediately decode/detach, and return CPU JVPs."""

    values = []
    for start in range(0, len(actions), outer_batch_size):
        stop = min(start + outer_batch_size, len(actions))
        jacobians = _transition_jacobians(
            model,
            initial[start:stop],
            suffix[start:stop],
            actions[start:stop],
            chunk_size=chunk_size,
        ).detach()
        decoded = torch.einsum("nda,na->nd", jacobians, delta[start:stop]) @ decoder
        values.append(decoded.detach().float().cpu())
        del decoded, jacobians
    return torch.cat(values)


def _composite_legendre(order: int, panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes/weights over equal panels partitioning [0, 1]."""

    base_nodes, base_weights = np.polynomial.legendre.leggauss(order)
    nodes = []
    weights = []
    width = 1.0 / panels
    for panel in range(panels):
        left = panel * width
        nodes.extend(left + (base_nodes + 1.0) * width / 2.0)
        weights.extend(base_weights * width / 2.0)
    return np.asarray(nodes), np.asarray(weights)


def _stratified_profile_indices(
    contexts: int, *, contexts_per_pair: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected = []
    pairs = []
    for pair_id in range(len(_action_pairs())):
        selected.extend(rng.choice(contexts, size=contexts_per_pair, replace=False))
        pairs.extend([pair_id] * contexts_per_pair)
    return np.asarray(selected, dtype=np.int64), np.asarray(pairs, dtype=np.int64)


def _stratified_permutation_null(
    predictor: np.ndarray,
    target: np.ndarray,
    strata: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(permutations):
        shuffled = predictor.copy()
        for stratum in np.unique(strata):
            mask = np.flatnonzero(strata == stratum)
            shuffled[mask] = shuffled[rng.permutation(mask)]
        values.append(_spearman(shuffled, target))
    array = np.asarray(values)
    return {
        "permutations": permutations,
        "mean_spearman": float(array.mean()),
        "p95_spearman": float(np.quantile(array, 0.95)),
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rank(left), _rank(right))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _decide(config: dict[str, Any], seed_results: list[dict[str, Any]]) -> dict[str, bool]:
    required = int(config.get("required_passing_seeds", 2))
    numerical = 0
    amplification = 0
    association = 0
    for seed in seed_results:
        horizons = seed["horizons"]
        numerical += int(
            all(
                horizons[str(h)]["summary"]["max_integration_relative_error"]
                <= float(config["integration_relative_error_max"])
                and horizons[str(h)]["summary"]["max_refinement_relative_change"]
                <= float(config["refinement_relative_change_max"])
                for h in (1, 4)
            )
        )
        amplification += int(
            seed["horizon4_to_horizon1_median_cancellation_ratio"]
            >= float(config["recurrent_cancellation_ratio_min"])
        )
        h4 = horizons["4"]["summary"]
        association += int(
            h4["cancellation_local_error_spearman"]
            >= float(config["cancellation_error_spearman_min"])
            and h4["cancellation_local_error_spearman"]
            >= h4["stratified_permutation_null"]["p95_spearman"]
            + float(config["permutation_margin_min"])
        )
    return {
        "h_geo_16_adaptive_path_fidelity": numerical >= required,
        "h_wm_15_recurrent_cancellation_amplification": amplification >= required,
        "h_wm_16_cancellation_predicts_local_failure": association >= required,
    }


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype(np.float64)


__all__ = [
    "_composite_legendre",
    "_decide",
    "_decoded_directional_derivatives",
    "_spearman",
    "_stratified_permutation_null",
    "_stratified_profile_indices",
    "run_lewm_action_path_geometry_study",
]
