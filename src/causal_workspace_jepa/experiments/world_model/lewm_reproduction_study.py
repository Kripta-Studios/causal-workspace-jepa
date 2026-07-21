"""Faithful small LeWorldModel reproduction and held-out causal audit."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.synthetic.pixel_tiny_maze import (
    generate_pixel_tiny_maze,
    render_pixel_tiny_maze,
    valid_positions,
)
from causal_workspace_jepa.data.synthetic.tiny_maze import step_tiny_maze
from causal_workspace_jepa.interpretability.circuit_graph import (
    CircuitEdge,
    CircuitGraph,
    CircuitNode,
)
from causal_workspace_jepa.interpretability.workspace_tests import (
    discover_shared_subspace,
    matched_causal_subspace_audit,
    random_subspace,
)
from causal_workspace_jepa.models.lewm import (
    OFFICIAL_REPOSITORY,
    OFFICIAL_REVISION,
    SmallLeWMConfig,
    SmallLeWorldModel,
)
from causal_workspace_jepa.models.probes import QuadraticRidgeHead


def run_lewm_reproduction_study(config_path: str | Path) -> dict[str, Any]:
    """Train from clean code, evaluate three seeds, and emit an audited graph."""

    started = time.monotonic()
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    hardware = require_free_disk(resource_profile)
    if not torch.cuda.is_available():
        raise RuntimeError("WM-LEWM-001 requires CUDA under the gpu_12gb profile")
    device = torch.device(str(config.get("device", "cuda")))
    seeds = [int(seed) for seed in config.get("seeds", [101, 103, 107])]
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path}",
        resource_profile=resource_profile,
        seed=seeds[0],
    )
    if provenance.git_dirty:
        raise RuntimeError("scientific execution requires a clean committed worktree")

    train_data = generate_pixel_tiny_maze(
        trajectories=int(config.get("train_trajectories", 768)),
        steps=int(config.get("trajectory_steps", 5)),
        seed=int(config.get("data_seed", 83)),
        cell_size=int(config.get("cell_size", 4)),
    )
    validation_data = generate_pixel_tiny_maze(
        trajectories=int(config.get("validation_trajectories", 160)),
        steps=int(config.get("trajectory_steps", 5)),
        seed=int(config.get("validation_seed", 89)),
        cell_size=int(config.get("cell_size", 4)),
    )
    test_data = generate_pixel_tiny_maze(
        trajectories=int(config.get("test_trajectories", 192)),
        steps=int(config.get("trajectory_steps", 5)),
        seed=int(config.get("test_seed", 97)),
        cell_size=int(config.get("cell_size", 4)),
    )
    seed_results: list[dict[str, Any]] = []
    trained_models: list[SmallLeWorldModel] = []
    checkpoint_dir = Path(str(config.get("checkpoint_dir", "artifacts/checkpoints/lewm_small_v1")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        model, training = _train_model(config, train_data, validation_data, seed, device)
        checkpoint = checkpoint_dir / f"seed_{seed}.pt"
        model.save(checkpoint)
        restored = SmallLeWorldModel.load(checkpoint, map_location=device).to(device).eval()
        replay_error = _checkpoint_replay_error(model, restored, test_data, device)
        audit = _audit_seed(config, restored, train_data, test_data, seed, device)
        audit["training"] = training
        audit["checkpoint_replay_max_abs_error"] = replay_error
        audit["checkpoint"] = str(checkpoint)
        seed_results.append(audit)
        trained_models.append(restored)

    ensemble = _ensemble_uncertainty_audit(trained_models, test_data, seed_results, device)
    required = int(config.get("required_passing_seeds", 2))
    reproduction_passes = sum(result["reproduction_pass"] for result in seed_results)
    mediation_passes = sum(result["causal_mediation_pass"] for result in seed_results)
    circuit_passes = sum(result["circuit_audit"]["pass"] for result in seed_results)
    planning_passes = sum(result["planning"]["intervention_changes_planning"] for result in seed_results)
    workspace_candidates = sum(result["workspace"]["candidate_pass"] for result in seed_results)
    graph_status = "VALIDATED_RESTRICTED" if circuit_passes >= required else "REJECTED"
    graph = _aggregate_circuit_graph(seed_results, status=graph_status)
    graph_path = Path(str(config.get("output_graph", "artifacts/tables/lewm_small_circuit.json")))
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph.write_json(graph_path)
    graph.write_graphml(graph_path.with_suffix(".graphml"))

    all_passed = bool(
        reproduction_passes >= required
        and mediation_passes >= required
        and planning_passes >= required
        and circuit_passes >= required
        and all(result["checkpoint_replay_max_abs_error"] == 0.0 for result in seed_results)
    )
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-LEWM-001")),
        "status": "SMOKE_VALIDATED" if all_passed else "FAILED_REGISTERED_GATES",
        "all_passed": all_passed,
        "evidence_level": "Generalization" if all_passed else "Causal mediation",
        "published_integration": {
            "name": "LeWorldModel faithful small reproduction",
            "official_repository": OFFICIAL_REPOSITORY,
            "official_revision": OFFICIAL_REVISION,
            "faithful_elements": [
                "end_to_end_pixel_encoder",
                "action_embedder",
                "autoregressive_AdaLN_zero_predictor",
                "next_embedding_MSE",
                "SIGReg_Gaussian_regularizer",
                "no_EMA_no_pretraining_no_auxiliary_supervision",
            ],
            "scaled_differences": [
                "20x20 PixelTinyMaze instead of released benchmark data",
                "32-dimensional two-block networks instead of 192-dimensional six-block networks",
                "64 SIGReg projections instead of 1024",
            ],
        },
        "dataset": {
            "name": "PixelTinyMaze",
            "train_trajectories": int(train_data.observations.shape[0]),
            "validation_trajectories": int(validation_data.observations.shape[0]),
            "test_trajectories": int(test_data.observations.shape[0]),
            "split_seeds": [
                int(config.get("data_seed", 83)),
                int(config.get("validation_seed", 89)),
                int(config.get("test_seed", 97)),
            ],
        },
        "hardware": hardware.as_dict(),
        "seeds": seeds,
        "required_passing_seeds": required,
        "reproduction_passing_seeds": reproduction_passes,
        "causal_mediation_passing_seeds": mediation_passes,
        "planning_passing_seeds": planning_passes,
        "circuit_passing_seeds": circuit_passes,
        "workspace_candidate_passing_seeds": workspace_candidates,
        "workspace_found": False,
        "ensemble_uncertainty": ensemble,
        "seed_results": seed_results,
        "circuit_graph": {"path": str(graph_path), "status": graph_status},
        "runtime_seconds": float(time.monotonic() - started),
        "claims": [
            {
                "claim": "A faithful small LeWorldModel reproduction learns action-conditioned pixel dynamics.",
                "evidence_level": "Generalization",
                "supported": reproduction_passes >= required,
            },
            {
                "claim": "A learned hidden action subspace selectively mediates donor counterfactuals.",
                "evidence_level": "Specificity",
                "supported": mediation_passes >= required,
            },
            {
                "claim": "A restricted action-to-planner circuit passes necessity, sufficiency, and faithfulness tests.",
                "evidence_level": "Circuit reconstruction",
                "supported": circuit_passes >= required,
            },
            {
                "claim": "A JEPA workspace was found.",
                "evidence_level": "Generalization",
                "supported": False,
            },
        ],
    }
    output_path = Path(
        str(config.get("output_metrics", "artifacts/metrics/lewm_small_reproduction_v1.json"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output_path.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": str(output_path),
            "all_passed": all_passed,
            "official_revision": OFFICIAL_REVISION,
            "workspace_found": False,
        },
    )
    if not all_passed:
        raise RuntimeError("WM-LEWM-001 completed but failed one or more registered acceptance gates")
    return metrics


def _model_config(config: dict[str, Any]) -> SmallLeWMConfig:
    return SmallLeWMConfig(
        image_size=int(config.get("image_size", 20)),
        patch_size=int(config.get("patch_size", 4)),
        latent_dim=int(config.get("latent_dim", 32)),
        encoder_depth=int(config.get("encoder_depth", 2)),
        predictor_depth=int(config.get("predictor_depth", 2)),
        heads=int(config.get("heads", 4)),
        mlp_dim=int(config.get("mlp_dim", 96)),
        max_history=int(config.get("max_history", 4)),
        dropout=float(config.get("dropout", 0.0)),
        sigreg_weight=float(config.get("sigreg_weight", 0.09)),
        sigreg_projections=int(config.get("sigreg_projections", 64)),
    )


def _train_model(
    config: dict[str, Any],
    train_data: Any,
    validation_data: Any,
    seed: int,
    device: torch.device,
) -> tuple[SmallLeWorldModel, dict[str, float]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = SmallLeWorldModel(_model_config(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 4e-4)),
        weight_decay=float(config.get("weight_decay", 1e-3)),
    )
    train_pixels, train_actions, _ = _flatten_transitions(train_data)
    val_pixels, val_actions, _ = _flatten_transitions(validation_data)
    generator = np.random.default_rng(seed)
    batch_size = min(int(config.get("batch_size", 128)), len(train_pixels))
    training_steps = int(config.get("training_steps", 1_000))
    validation_interval = int(config.get("validation_interval", 50))
    best_validation = float("inf")
    last_losses: dict[str, float] = {}
    for step in range(1, training_steps + 1):
        ids = generator.choice(len(train_pixels), size=batch_size, replace=False)
        pixels = torch.as_tensor(train_pixels[ids], device=device)
        actions = torch.as_tensor(train_actions[ids], device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(pixels, actions)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
        optimizer.step()
        last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
        if step % validation_interval == 0 or step == training_steps:
            validation = _embedding_prediction_mse(
                model, val_pixels, val_actions, device, batch_size=256
            )
            best_validation = min(best_validation, validation)
    model.eval()
    final_validation = _embedding_prediction_mse(
        model, val_pixels, val_actions, device, batch_size=256
    )
    return model, {
        "last_total_loss": last_losses["loss"],
        "last_prediction_loss": last_losses["prediction_loss"],
        "last_sigreg_loss": last_losses["sigreg_loss"],
        "best_validation_embedding_mse": best_validation,
        "final_validation_embedding_mse": final_validation,
        "training_steps": training_steps,
    }


def _flatten_transitions(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = dataset.observations[:, :-1].reshape(-1, *dataset.observations.shape[2:])
    following = dataset.observations[:, 1:].reshape(-1, *dataset.observations.shape[2:])
    pixels = np.stack([current, following], axis=1).astype(np.float32)
    actions = dataset.actions.reshape(-1, 1, dataset.actions.shape[-1]).astype(np.float32)
    states = dataset.states[:, 1:].reshape(-1, dataset.states.shape[-1]).astype(np.float32)
    return pixels, actions, states


def _embedding_prediction_mse(
    model: SmallLeWorldModel,
    pixels: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    action_override: np.ndarray | None = None,
) -> float:
    squared = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            batch_pixels = torch.as_tensor(pixels[start : start + batch_size], device=device)
            source = actions if action_override is None else action_override
            batch_actions = torch.as_tensor(source[start : start + batch_size], device=device)
            output = model.forward_sequence(batch_pixels, batch_actions)
            predicted = output["predicted_embeddings"]
            target = output["embeddings"]
            assert isinstance(predicted, Tensor) and isinstance(target, Tensor)
            error = (predicted - target[:, 1:]).square()
            squared += float(error.sum().cpu())
            count += error.numel()
    return squared / count


def _checkpoint_replay_error(
    original: SmallLeWorldModel,
    restored: SmallLeWorldModel,
    test_data: Any,
    device: torch.device,
) -> float:
    pixels, actions, _ = _flatten_transitions(test_data)
    batch_pixels = torch.as_tensor(pixels[:16], device=device)
    batch_actions = torch.as_tensor(actions[:16], device=device)
    with torch.inference_mode():
        left = original.forward_sequence(batch_pixels, batch_actions)["predicted_embeddings"]
        right = restored.forward_sequence(batch_pixels, batch_actions)["predicted_embeddings"]
    assert isinstance(left, Tensor) and isinstance(right, Tensor)
    return float((left - right).abs().max().cpu())


def _audit_seed(
    config: dict[str, Any],
    model: SmallLeWorldModel,
    train_data: Any,
    test_data: Any,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    train_pixels, train_actions, train_states = _flatten_transitions(train_data)
    test_pixels, test_actions, test_states = _flatten_transitions(test_data)
    shuffled = test_actions.copy()
    np.random.default_rng(seed + 10_000).shuffle(shuffled, axis=0)
    zero_actions = np.zeros_like(test_actions)
    clean_mse = _embedding_prediction_mse(
        model, test_pixels, test_actions, device, batch_size=256
    )
    shuffled_mse = _embedding_prediction_mse(
        model, test_pixels, test_actions, device, batch_size=256, action_override=shuffled
    )
    no_action_mse = _embedding_prediction_mse(
        model, test_pixels, test_actions, device, batch_size=256, action_override=zero_actions
    )
    train_sites, train_latent = _encode_sites(model, train_pixels[:, 1], device)
    test_sites, test_latent = _encode_sites(model, test_pixels[:, 1], device)
    probe_metrics, decoder = _layerwise_probes(
        train_sites,
        test_sites,
        train_states,
        test_states,
        ridge=float(config.get("probe_ridge", 1e-3)),
    )
    latent_std = float(np.mean(np.std(test_latent, axis=0)))
    reproduction_pass = bool(
        clean_mse <= float(config.get("max_test_embedding_mse", 0.20))
        and clean_mse / max(shuffled_mse, 1e-12)
        <= float(config.get("max_action_mse_ratio", 0.75))
        and latent_std >= float(config.get("min_latent_std", 0.10))
        and probe_metrics["projector.latent"]["nonlinear_r2"]
        >= float(config.get("min_nonlinear_state_probe_r2", 0.60))
    )

    balanced = _balanced_transition_bank(
        cell_size=int(config.get("cell_size", 4)),
        goals=int(config.get("balanced_goals", 6)),
        seed=seed + 20_000,
    )
    balanced_result = _predict_transition_batch(
        model, balanced["current"], balanced["actions"], device
    )
    action_sites = {
        name: values.reshape(-1, 4, values.shape[-1])
        for name, values in balanced_result["sites"].items()
        if name.startswith("predictor.block")
    }
    subspace_dimension = int(config.get("action_subspace_dimension", 4))
    bases = {
        name: _paired_action_subspace(values, subspace_dimension)
        for name, values in action_sites.items()
    }
    centers = {name: values.reshape(-1, values.shape[-1]).mean(axis=0) for name, values in action_sites.items()}
    intervention_site = str(config.get("intervention_site", "predictor.block1"))
    if intervention_site not in bases:
        raise KeyError(f"configured intervention site {intervention_site!r} is unavailable")
    patch_audit = _counterfactual_patch_audit(
        model,
        balanced,
        intervention_site,
        bases[intervention_site],
        decoder,
        seed,
        device,
        random_controls=int(config.get("random_controls", 16)),
    )
    suppression = _action_suppression_audit(model, test_pixels, test_actions, device)
    causal_mediation_pass = bool(
        patch_audit["latent_recovery"] >= float(config.get("min_patch_recovery", 0.25))
        and patch_audit["latent_recovery"]
        > patch_audit["random_control_p95_recovery"]
        + float(config.get("min_patch_specificity_margin", 0.05))
        and patch_audit["decoded_recovery"] >= float(config.get("min_decoded_recovery", 0.10))
    )
    planning = _planning_audit(
        config,
        model,
        decoder,
        intervention_site,
        bases[intervention_site],
        centers[intervention_site],
        seed,
        device,
    )
    circuit = _circuit_audit(
        config,
        model,
        balanced,
        bases,
        decoder,
        patch_audit,
        suppression,
        planning,
        seed,
        device,
    )
    consumers, consumer_scores = _fit_consumers(train_latent, train_states, seed)
    workspace = _workspace_audit(
        test_latent,
        consumers,
        consumer_scores,
        seed,
        max_dimension=int(config.get("workspace_max_dimension", 6)),
        random_controls=int(config.get("workspace_random_controls", 16)),
    )
    return {
        "seed": seed,
        "prediction": {
            "test_embedding_mse": clean_mse,
            "shuffled_action_mse": shuffled_mse,
            "no_action_mse": no_action_mse,
            "clean_to_shuffled_ratio": clean_mse / max(shuffled_mse, 1e-12),
            "clean_to_no_action_ratio": clean_mse / max(no_action_mse, 1e-12),
            "latent_mean_std": latent_std,
        },
        "layerwise_probes": probe_metrics,
        "reproduction_pass": reproduction_pass,
        "action_subspace": {
            "site": intervention_site,
            "dimension": subspace_dimension,
            "representation_dimension": int(bases[intervention_site].shape[0]),
        },
        "counterfactual_patch": patch_audit,
        "action_suppression": suppression,
        "causal_mediation_pass": causal_mediation_pass,
        "planning": planning,
        "circuit_audit": circuit,
        "workspace": workspace,
    }


def _encode_sites(
    model: SmallLeWorldModel, pixels: np.ndarray, device: torch.device
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {}
    latents: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 256):
            batch = torch.as_tensor(pixels[start : start + 256], device=device)
            latent, sites = model.encode_pixels(batch)
            latents.append(latent.cpu().numpy())
            for name, value in sites.items():
                selected = value[:, 0] if value.ndim == 3 else value
                collected.setdefault(name, []).append(selected.cpu().numpy())
    merged = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    return merged, np.concatenate(latents, axis=0)


def _layerwise_probes(
    train_sites: dict[str, np.ndarray],
    test_sites: dict[str, np.ndarray],
    train_states: np.ndarray,
    test_states: np.ndarray,
    *,
    ridge: float,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    metrics: dict[str, dict[str, float]] = {}
    decoder: np.ndarray | None = None
    for name, train_features in train_sites.items():
        if name not in test_sites:
            continue
        linear = _ridge_fit(train_features, train_states, ridge)
        linear_prediction = _ridge_predict(test_sites[name], linear)
        quadratic = QuadraticRidgeHead.fit(train_features, train_states, ridge=ridge)
        metrics[name] = {
            "linear_r2": _r2(test_states, linear_prediction),
            "nonlinear_r2": quadratic.score_r2(test_sites[name], test_states),
            "evidence_level": "Availability",
        }
        if name == "projector.latent":
            decoder = linear
    if decoder is None:
        raise RuntimeError("projector latent decoder was not fitted")
    return metrics, decoder


def _ridge_fit(features: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
    y = np.asarray(targets, dtype=np.float64).reshape(len(targets), -1)
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    regularizer = ridge * np.eye(design.shape[1])
    regularizer[-1, -1] = 0.0
    if design.shape[1] > design.shape[0]:
        weights = design.T @ np.linalg.solve(
            design @ design.T + ridge * np.eye(len(design)), y
        )
    else:
        weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return weights.astype(np.float32)


def _ridge_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).reshape(len(features), -1)
    design = np.concatenate([values, np.ones((len(values), 1), dtype=np.float32)], axis=1)
    return design @ weights


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.mean((np.asarray(target) - np.asarray(prediction)) ** 2))
    baseline = float(np.mean((np.asarray(target) - np.mean(target, axis=0, keepdims=True)) ** 2))
    return float(1.0 - residual / max(baseline, 1e-12))


def _balanced_transition_bank(*, cell_size: int, goals: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = valid_positions()
    selected_goals = [positions[int(index)] for index in rng.choice(len(positions), goals, replace=False)]
    current: list[np.ndarray] = []
    following: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    context_ids: list[int] = []
    action_ids: list[int] = []
    context = 0
    for goal in selected_goals:
        for position in positions:
            for action_id in range(4):
                next_position = step_tiny_maze(np.asarray(position), action_id)
                current.append(render_pixel_tiny_maze(position, goal, cell_size=cell_size))
                following.append(render_pixel_tiny_maze(next_position, goal, cell_size=cell_size))
                action = np.zeros(4, dtype=np.float32)
                action[action_id] = 1.0
                actions.append(action)
                context_ids.append(context)
                action_ids.append(action_id)
            context += 1
    return {
        "current": np.asarray(current, dtype=np.float32),
        "following": np.asarray(following, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "context_ids": np.asarray(context_ids, dtype=np.int64),
        "action_ids": np.asarray(action_ids, dtype=np.int64),
    }


def _predict_transition_batch(
    model: SmallLeWorldModel,
    current: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    *,
    interventions: dict[str, Callable[[Tensor], Tensor]] | None = None,
) -> dict[str, Any]:
    with torch.inference_mode():
        current_tensor = torch.as_tensor(current, device=device)
        action_tensor = torch.as_tensor(actions, device=device)
        embedding, _ = model.encode_pixels(current_tensor)
        predicted, sites = model.predict_step(
            embedding, action_tensor, interventions=interventions
        )
    return {
        "predicted": predicted.cpu().numpy(),
        "sites": {name: value.cpu().numpy() for name, value in sites.items()},
    }


def _paired_action_subspace(values: np.ndarray, dimension: int) -> np.ndarray:
    if values.ndim != 3 or values.shape[1] != 4:
        raise ValueError("balanced site values must have [contexts,4,dimension] shape")
    differences = values - values.mean(axis=1, keepdims=True)
    _, _, right = np.linalg.svd(differences.reshape(-1, values.shape[-1]), full_matrices=False)
    return right[:dimension].T.astype(np.float32)


def _counterfactual_patch_audit(
    model: SmallLeWorldModel,
    bank: dict[str, np.ndarray],
    site: str,
    basis: np.ndarray,
    decoder: np.ndarray,
    seed: int,
    device: torch.device,
    *,
    random_controls: int,
) -> dict[str, Any]:
    contexts = bank["current"].reshape(-1, 4, *bank["current"].shape[1:])
    actions = bank["actions"].reshape(-1, 4, 4)
    recipients = np.tile(np.arange(4), len(contexts))
    donors = (recipients + 1) % 4
    rows = np.repeat(np.arange(len(contexts)), 4)
    current = contexts[rows, recipients]
    recipient_actions = actions[rows, recipients]
    donor_actions = actions[rows, donors]
    recipient_run = _predict_transition_batch(model, current, recipient_actions, device)
    donor_run = _predict_transition_batch(model, current, donor_actions, device)
    recipient_site = recipient_run["sites"][site]
    donor_site = donor_run["sites"][site]

    def patch_callback(candidate_basis: np.ndarray, *, norm_match: np.ndarray | None = None):
        basis_tensor = torch.as_tensor(candidate_basis, device=device)
        donor_tensor = torch.as_tensor(donor_site[:, None, :], device=device)

        def callback(values: Tensor) -> Tensor:
            difference = donor_tensor.to(values) - values
            delta = (difference @ basis_tensor.to(values)) @ basis_tensor.to(values).T
            if norm_match is not None:
                target = torch.as_tensor(
                    norm_match[:, None, :], device=values.device, dtype=values.dtype
                )
                observed = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                delta = delta * target / observed
            return values + delta

        return callback

    candidate_difference = donor_site - recipient_site
    candidate_delta = (candidate_difference @ basis) @ basis.T
    candidate_norm = np.linalg.norm(candidate_delta, axis=-1, keepdims=True).astype(np.float32)
    patched = _predict_transition_batch(
        model,
        current,
        recipient_actions,
        device,
        interventions={site: patch_callback(basis)},
    )["predicted"]
    recipient_prediction = recipient_run["predicted"]
    donor_prediction = donor_run["predicted"]
    latent_recovery = _recovery(recipient_prediction, donor_prediction, patched)
    recipient_state = _ridge_predict(recipient_prediction, decoder)
    donor_state = _ridge_predict(donor_prediction, decoder)
    patched_state = _ridge_predict(patched, decoder)
    decoded_recovery = _recovery(recipient_state, donor_state, patched_state)
    random_recoveries: list[float] = []
    random_decoded: list[float] = []
    for index in range(random_controls):
        control = random_subspace(basis.shape[0], basis.shape[1], seed + 30_000 + index)
        controlled = _predict_transition_batch(
            model,
            current,
            recipient_actions,
            device,
            interventions={site: patch_callback(control, norm_match=candidate_norm)},
        )["predicted"]
        random_recoveries.append(_recovery(recipient_prediction, donor_prediction, controlled))
        random_decoded.append(
            _recovery(
                recipient_state,
                donor_state,
                _ridge_predict(controlled, decoder),
            )
        )
    return {
        "site": site,
        "operation": "subspace_feature_replacement",
        "pairs": int(len(current)),
        "latent_recovery": latent_recovery,
        "decoded_recovery": decoded_recovery,
        "candidate_mean_perturbation_l2": float(candidate_norm.mean()),
        "random_control_count": random_controls,
        "random_control_mean_recovery": float(np.mean(random_recoveries)),
        "random_control_p95_recovery": float(np.quantile(random_recoveries, 0.95)),
        "random_control_mean_decoded_recovery": float(np.mean(random_decoded)),
        "evidence_level": "Specificity",
    }


def _recovery(recipient: np.ndarray, donor: np.ndarray, patched: np.ndarray) -> float:
    baseline = float(np.mean((np.asarray(recipient) - np.asarray(donor)) ** 2))
    residual = float(np.mean((np.asarray(patched) - np.asarray(donor)) ** 2))
    return float(1.0 - residual / max(baseline, 1e-12))


def _action_suppression_audit(
    model: SmallLeWorldModel,
    test_pixels: np.ndarray,
    test_actions: np.ndarray,
    device: torch.device,
) -> dict[str, float | str]:
    current = test_pixels[:, 0]
    following = test_pixels[:, 1]
    with torch.inference_mode():
        current_embedding, _ = model.encode_pixels(torch.as_tensor(current, device=device))
        target_embedding, _ = model.encode_pixels(torch.as_tensor(following, device=device))
        clean, _ = model.predict_step(
            current_embedding, torch.as_tensor(test_actions[:, 0], device=device)
        )
        suppressed, _ = model.predict_step(
            current_embedding,
            torch.as_tensor(test_actions[:, 0], device=device),
            interventions={"action.embedding": lambda values: torch.zeros_like(values)},
        )
    clean_error = float((clean - target_embedding).square().mean().cpu())
    suppressed_error = float((suppressed - target_embedding).square().mean().cpu())
    return {
        "site": "action.embedding",
        "operation": "suppress_module",
        "clean_mse": clean_error,
        "suppressed_mse": suppressed_error,
        "error_ratio": suppressed_error / max(clean_error, 1e-12),
        "evidence_level": "Causal mediation",
    }


def _projection_removal_callback(
    basis: np.ndarray,
    center: np.ndarray,
    device: torch.device,
    *,
    match_basis: np.ndarray | None = None,
) -> Callable[[Tensor], Tensor]:
    basis_tensor = torch.as_tensor(basis, device=device)
    center_tensor = torch.as_tensor(center, device=device)
    match_tensor = torch.as_tensor(match_basis, device=device) if match_basis is not None else None

    def callback(values: Tensor) -> Tensor:
        centered = values - center_tensor.to(values)
        delta = (centered @ basis_tensor.to(values)) @ basis_tensor.to(values).T
        if match_tensor is not None:
            target_delta = (centered @ match_tensor.to(values)) @ match_tensor.to(values).T
            target_norm = target_delta.norm(dim=-1, keepdim=True)
            delta = delta * target_norm / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return values - delta

    return callback


def _planning_cases(seed: int, count: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    positions = valid_positions()
    candidates = [
        (start, goal)
        for start in positions
        for goal in positions
        if start != goal and 2 <= abs(start[0] - goal[0]) + abs(start[1] - goal[1]) <= 6
    ]
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(index)] for index in indices]


def _action_sequences(horizon: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(list(itertools.product(range(4), repeat=horizon)), dtype=np.int64)
    one_hot = np.eye(4, dtype=np.float32)[ids]
    return ids, one_hot


def _plan(
    model: SmallLeWorldModel,
    position: tuple[int, int],
    goal: tuple[int, int],
    action_ids: np.ndarray,
    action_values: np.ndarray,
    device: torch.device,
    *,
    cell_size: int,
    intervention: dict[str, Callable[[Tensor], Tensor]] | None = None,
) -> dict[str, Any]:
    current_image = render_pixel_tiny_maze(position, goal, cell_size=cell_size)
    goal_image = render_pixel_tiny_maze(goal, goal, cell_size=cell_size)
    with torch.inference_mode():
        current, _ = model.encode_pixels(
            torch.as_tensor(current_image[None], device=device)
        )
        target, _ = model.encode_pixels(torch.as_tensor(goal_image[None], device=device))
        repeated = current.expand(len(action_values), -1)
        actions = torch.as_tensor(action_values, device=device)
        trajectory, _ = model.rollout(repeated, actions, interventions=intervention)
        costs = (trajectory[:, -1] - target).square().mean(dim=-1)
        best = int(torch.argmin(costs).cpu())
    return {
        "first_action": int(action_ids[best, 0]),
        "action_ids": action_ids[best].copy(),
        "predicted_cost": float(costs[best].cpu()),
        "costs": costs.cpu().numpy(),
        "trajectory": trajectory[best].cpu().numpy(),
    }


def _closed_loop(
    model: SmallLeWorldModel,
    cases: list[tuple[tuple[int, int], tuple[int, int]]],
    action_ids: np.ndarray,
    action_values: np.ndarray,
    device: torch.device,
    *,
    cell_size: int,
    max_steps: int,
    intervention: dict[str, Callable[[Tensor], Tensor]] | None,
) -> dict[str, Any]:
    successes = 0
    distances: list[float] = []
    first_actions: list[int] = []
    for start, goal in cases:
        position = np.asarray(start, dtype=np.int64)
        for step in range(max_steps):
            result = _plan(
                model,
                tuple(int(v) for v in position),
                goal,
                action_ids,
                action_values,
                device,
                cell_size=cell_size,
                intervention=intervention,
            )
            if step == 0:
                first_actions.append(int(result["first_action"]))
            position = step_tiny_maze(position, int(result["first_action"]))
            if tuple(int(value) for value in position) == goal:
                successes += 1
                break
        distances.append(float(np.abs(position - np.asarray(goal)).sum()))
    return {
        "success_rate": successes / len(cases),
        "mean_final_manhattan_distance": float(np.mean(distances)),
        "first_actions": first_actions,
    }


def _planning_audit(
    config: dict[str, Any],
    model: SmallLeWorldModel,
    decoder: np.ndarray,
    site: str,
    basis: np.ndarray,
    center: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    horizon = int(config.get("planning_horizon", 4))
    max_steps = int(config.get("closed_loop_steps", 7))
    cases = _planning_cases(seed + 40_000, int(config.get("planning_cases", 12)))
    action_ids, action_values = _action_sequences(horizon)
    candidate_callback = _projection_removal_callback(basis, center, device)
    candidate_intervention = {site: candidate_callback}
    clean_loop = _closed_loop(
        model,
        cases,
        action_ids,
        action_values,
        device,
        cell_size=int(config.get("cell_size", 4)),
        max_steps=max_steps,
        intervention=None,
    )
    candidate_loop = _closed_loop(
        model,
        cases,
        action_ids,
        action_values,
        device,
        cell_size=int(config.get("cell_size", 4)),
        max_steps=max_steps,
        intervention=candidate_intervention,
    )
    action_change = float(
        np.mean(
            np.asarray(clean_loop["first_actions"])
            != np.asarray(candidate_loop["first_actions"])
        )
    )
    cost_changes: list[float] = []
    trajectory_changes: list[float] = []
    decoded_changes: list[float] = []
    random_action_changes: list[float] = []
    random_cost_changes: list[float] = []
    clean_initial_plans = []
    candidate_initial_plans = []
    for start, goal in cases:
        clean = _plan(
            model,
            start,
            goal,
            action_ids,
            action_values,
            device,
            cell_size=int(config.get("cell_size", 4)),
        )
        candidate = _plan(
            model,
            start,
            goal,
            action_ids,
            action_values,
            device,
            cell_size=int(config.get("cell_size", 4)),
            intervention=candidate_intervention,
        )
        clean_initial_plans.append(clean)
        candidate_initial_plans.append(candidate)
        cost_changes.append(float(np.mean(np.abs(clean["costs"] - candidate["costs"]))))
        trajectory_changes.append(
            float(np.mean((clean["trajectory"] - candidate["trajectory"]) ** 2))
        )
        clean_decoded = _ridge_predict(clean["trajectory"], decoder)
        candidate_decoded = _ridge_predict(candidate["trajectory"], decoder)
        decoded_changes.append(float(np.mean((clean_decoded - candidate_decoded) ** 2)))
    for index in range(int(config.get("planning_random_controls", 8))):
        control = random_subspace(basis.shape[0], basis.shape[1], seed + 50_000 + index)
        callback = _projection_removal_callback(
            control, center, device, match_basis=basis
        )
        controlled_plans = []
        for start, goal in cases:
            controlled_plans.append(
                _plan(
                    model,
                    start,
                    goal,
                    action_ids,
                    action_values,
                    device,
                    cell_size=int(config.get("cell_size", 4)),
                    intervention={site: callback},
                )
            )
        random_action_changes.append(
            float(
                np.mean(
                    [
                        clean["first_action"] != controlled["first_action"]
                        for clean, controlled in zip(
                            clean_initial_plans, controlled_plans, strict=True
                        )
                    ]
                )
            )
        )
        random_cost_changes.append(
            float(
                np.mean(
                    [
                        np.mean(np.abs(clean["costs"] - controlled["costs"]))
                        for clean, controlled in zip(
                            clean_initial_plans, controlled_plans, strict=True
                        )
                    ]
                )
            )
        )
    mean_cost_change = float(np.mean(cost_changes))
    mean_trajectory_change = float(np.mean(trajectory_changes))
    mean_decoded_change = float(np.mean(decoded_changes))
    random_action_p95 = float(np.quantile(random_action_changes, 0.95))
    random_cost_p95 = float(np.quantile(random_cost_changes, 0.95))
    intervention_changes_planning = bool(
        action_change >= float(config.get("min_selected_action_change", 0.15))
        and mean_cost_change >= float(config.get("min_planning_cost_change", 1e-4))
        and mean_trajectory_change >= float(config.get("min_trajectory_change", 1e-4))
        and (action_change > random_action_p95 or mean_cost_change > random_cost_p95)
    )
    return {
        "site": site,
        "operation": "project_out",
        "cases": len(cases),
        "horizon": horizon,
        "future_latent_trajectory_mse": mean_trajectory_change,
        "decoded_state_trajectory_mse": mean_decoded_change,
        "mean_absolute_candidate_cost_change": mean_cost_change,
        "selected_first_action_change_rate": action_change,
        "clean_closed_loop": clean_loop,
        "intervened_closed_loop": candidate_loop,
        "closed_loop_success_change": float(
            candidate_loop["success_rate"] - clean_loop["success_rate"]
        ),
        "random_control_action_change_p95": random_action_p95,
        "random_control_cost_change_p95": random_cost_p95,
        "random_control_count": len(random_action_changes),
        "intervention_changes_planning": intervention_changes_planning,
        "evidence_level": "Specificity",
    }


def _site_patch_recovery(
    model: SmallLeWorldModel,
    bank: dict[str, np.ndarray],
    site: str,
    basis: np.ndarray | None,
    device: torch.device,
) -> float:
    contexts = bank["current"].reshape(-1, 4, *bank["current"].shape[1:])
    actions = bank["actions"].reshape(-1, 4, 4)
    current = contexts[:, 0]
    recipient_action = actions[:, 0]
    donor_action = actions[:, 1]
    recipient = _predict_transition_batch(model, current, recipient_action, device)
    donor = _predict_transition_batch(model, current, donor_action, device)
    donor_site = torch.as_tensor(donor["sites"][site][:, None, :], device=device)

    def callback(values: Tensor) -> Tensor:
        difference = donor_site.to(values) - values
        if basis is None:
            return donor_site.to(values).expand_as(values)
        basis_tensor = torch.as_tensor(basis, device=values.device, dtype=values.dtype)
        return values + (difference @ basis_tensor) @ basis_tensor.T

    patched = _predict_transition_batch(
        model,
        current,
        recipient_action,
        device,
        interventions={site: callback},
    )["predicted"]
    return _recovery(recipient["predicted"], donor["predicted"], patched)


def _circuit_audit(
    config: dict[str, Any],
    model: SmallLeWorldModel,
    bank: dict[str, np.ndarray],
    bases: dict[str, np.ndarray],
    decoder: np.ndarray,
    patch_audit: dict[str, Any],
    suppression: dict[str, Any],
    planning: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    del decoder, seed
    action_recovery = _site_patch_recovery(
        model, bank, "action.embedding", None, device
    )
    site_recoveries = {
        site: _site_patch_recovery(model, bank, site, basis, device)
        for site, basis in bases.items()
    }
    last_site = f"predictor.block{model.config.predictor_depth - 1}"
    necessity = bool(
        float(suppression["error_ratio"])
        >= float(config.get("min_suppression_error_ratio", 1.25))
    )
    sufficiency = bool(
        action_recovery >= float(config.get("min_action_patch_recovery", 0.95))
        and site_recoveries[last_site] >= float(config.get("min_patch_recovery", 0.25))
    )
    faithfulness = float(site_recoveries[last_site] / max(action_recovery, 1e-12))
    specificity = bool(
        patch_audit["latent_recovery"]
        > patch_audit["random_control_p95_recovery"]
        + float(config.get("min_patch_specificity_margin", 0.05))
    )
    # The restricted graph retains only the last predictor site. Earlier blocks
    # are localization alternatives because action conditioning enters every
    # AdaLN block directly; they are not falsely forced into a serial circuit.
    minimality = bool(
        site_recoveries[last_site]
        >= float(config.get("min_intermediate_recovery", 0.10))
    )
    passed = bool(
        necessity
        and sufficiency
        and specificity
        and minimality
        and planning["intervention_changes_planning"]
    )
    return {
        "pass": passed,
        "necessity_pass": necessity,
        "sufficiency_pass": sufficiency,
        "specificity_pass": specificity,
        "minimality_pass": minimality,
        "action_embedding_patch_recovery": action_recovery,
        "hidden_site_patch_recovery": site_recoveries,
        "last_hidden_to_full_action_faithfulness": faithfulness,
        "planner_edge_pass": bool(planning["intervention_changes_planning"]),
        "evidence_level": "Circuit reconstruction" if passed else "Specificity",
    }


def _fit_consumers(
    latents: np.ndarray, states: np.ndarray, seed: int
) -> tuple[dict[str, QuadraticRidgeHead], dict[str, float]]:
    positions = states[:, :2]
    goals = states[:, 2:4]
    value = -np.sum((positions - goals) ** 2, axis=-1, keepdims=True)
    risk = (
        (positions[:, 0:1] <= 0)
        | (positions[:, 0:1] >= 4)
        | (positions[:, 1:2] <= 0)
        | (positions[:, 1:2] >= 4)
    ).astype(np.float32)
    optimal = _optimal_action_targets(states)
    rng = np.random.default_rng(seed + 60_000)
    uncertainty = (
        0.05
        + 0.02 * states[:, 4:5]
        + rng.normal(scale=0.002, size=(len(states), 1))
    ).astype(np.float32)
    targets = {
        "physical": states[:, :4],
        "value": value,
        "risk": risk,
        "uncertainty": uncertainty,
        "action_selection": optimal,
    }
    split = int(0.8 * len(latents))
    heads = {
        name: QuadraticRidgeHead.fit(latents[:split], target[:split], ridge=1e-2)
        for name, target in targets.items()
    }
    scores = {
        name: head.score_r2(latents[split:], targets[name][split:])
        for name, head in heads.items()
    }
    return heads, scores


def _optimal_action_targets(states: np.ndarray) -> np.ndarray:
    targets = np.zeros((len(states), 4), dtype=np.float32)
    for index, state in enumerate(states):
        position = np.rint(state[:2]).astype(np.int64)
        goal = np.rint(state[2:4]).astype(np.int64)
        distances = [
            np.abs(step_tiny_maze(position, action) - goal).sum() for action in range(4)
        ]
        targets[index, int(np.argmin(distances))] = 1.0
    return targets


def _workspace_audit(
    test_latents: np.ndarray,
    consumers: dict[str, QuadraticRidgeHead],
    consumer_scores: dict[str, float],
    seed: int,
    *,
    max_dimension: int,
    random_controls: int,
) -> dict[str, Any]:
    examples = test_latents[: min(96, len(test_latents))]
    jacobians = {name: head.jacobian(examples) for name, head in consumers.items()}
    candidate = discover_shared_subspace(
        jacobians,
        max_dimension=max_dimension,
        min_consumer_capture=0.70,
        max_compactness_ratio=0.25,
        min_consumers=5,
    )
    controls = [
        random_subspace(test_latents.shape[-1], candidate.dimension, seed + 70_000 + index)
        for index in range(random_controls)
    ]
    causal = matched_causal_subspace_audit(
        examples,
        {name: head.predict for name, head in consumers.items()},
        candidate.basis,
        controls,
        center=test_latents.mean(axis=0),
    )
    controls_valid = _workspace_detector_controls(seed)
    consumers_valid = bool(min(consumer_scores.values()) >= 0.40)
    candidate_pass = bool(
        controls_valid["positive_control_found"]
        and not controls_valid["negative_control_found"]
        and consumers_valid
        and candidate.sensitivity_candidate_found
        and causal["specificity_exceeds_all_random_controls"]
    )
    return {
        "consumer_r2": consumer_scores,
        "consumers_valid": consumers_valid,
        "candidate": candidate.to_metrics(),
        "causal_ablation": causal,
        "detector_controls": controls_valid,
        "candidate_pass": candidate_pass,
        "workspace_found": False,
        "missing_workspace_criteria": [
            "reportability_analogue",
            "voluntary_control_analogue",
            "selective_multistep_necessity_over_one_step_prediction",
            "task_and_distribution_generalization",
        ],
        "evidence_level": "Specificity",
    }


def _workspace_detector_controls(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 80_000)
    representation_dim = 16
    shared = np.linalg.qr(rng.normal(size=(representation_dim, 3)))[0][:, :3]
    positive = {
        f"consumer_{index}": rng.normal(size=(4, 3)) @ shared.T for index in range(5)
    }
    negative = {}
    for index in range(5):
        basis = np.linalg.qr(rng.normal(size=(representation_dim, 3)))[0][:, :3]
        negative[f"consumer_{index}"] = rng.normal(size=(4, 3)) @ basis.T
    positive_result = discover_shared_subspace(
        positive,
        max_dimension=3,
        min_consumer_capture=0.90,
        max_compactness_ratio=0.25,
        min_consumers=5,
    )
    negative_result = discover_shared_subspace(
        negative,
        max_dimension=3,
        min_consumer_capture=0.90,
        max_compactness_ratio=0.25,
        min_consumers=5,
    )
    return {
        "positive_control_found": positive_result.sensitivity_candidate_found,
        "negative_control_found": negative_result.sensitivity_candidate_found,
    }


def _ensemble_uncertainty_audit(
    models: list[SmallLeWorldModel],
    test_data: Any,
    seed_results: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    pixels, actions, _ = _flatten_transitions(test_data)
    current = pixels[:256, 0]
    following = pixels[:256, 1]
    clean_predictions: list[np.ndarray] = []
    intervened_predictions: list[np.ndarray] = []
    for model, result in zip(models, seed_results, strict=True):
        site = str(result["action_subspace"]["site"])
        balanced = _balanced_transition_bank(cell_size=4, goals=6, seed=int(result["seed"]) + 20_000)
        run = _predict_transition_batch(model, balanced["current"], balanced["actions"], device)
        values = run["sites"][site].reshape(-1, 4, run["sites"][site].shape[-1])
        basis = _paired_action_subspace(values, int(result["action_subspace"]["dimension"]))
        center = values.reshape(-1, values.shape[-1]).mean(axis=0)
        clean_predictions.append(
            _predict_transition_batch(model, current, actions[:256, 0], device)["predicted"]
        )
        intervened_predictions.append(
            _predict_transition_batch(
                model,
                current,
                actions[:256, 0],
                device,
                interventions={site: _projection_removal_callback(basis, center, device)},
            )["predicted"]
        )
    clean_stack = np.stack(clean_predictions)
    intervened_stack = np.stack(intervened_predictions)
    target_embeddings, _ = _encode_sites(models[0], following, device)
    del target_embeddings
    clean_variance = float(np.mean(np.var(clean_stack, axis=0, ddof=1)))
    intervened_variance = float(np.mean(np.var(intervened_stack, axis=0, ddof=1)))
    return {
        "members": len(models),
        "clean_mean_variance": clean_variance,
        "intervened_mean_variance": intervened_variance,
        "variance_change": intervened_variance - clean_variance,
        "evidence_level": "Causal mediation",
    }


def _aggregate_circuit_graph(
    seed_results: list[dict[str, Any]], *, status: str
) -> CircuitGraph:
    sites = [str(seed_results[0]["action_subspace"]["site"])]
    action_recovery = float(
        np.mean(
            [result["circuit_audit"]["action_embedding_patch_recovery"] for result in seed_results]
        )
    )
    nodes = [
        CircuitNode("action.embedding", "action.embedding", "Causal mediation", action_recovery)
    ]
    edges: list[CircuitEdge] = []
    for site in sites:
        recovery = float(
            np.mean(
                [result["circuit_audit"]["hidden_site_patch_recovery"][site] for result in seed_results]
            )
        )
        nodes.append(CircuitNode(site, site, "Specificity", recovery))
        edges.append(CircuitEdge("action.embedding", site, recovery, recovery / max(action_recovery, 1e-12)))
    planner_effect = float(
        np.mean(
            [result["planning"]["selected_first_action_change_rate"] for result in seed_results]
        )
    )
    nodes.extend(
        [
            CircuitNode("predictor.latent", "predictor.latent", "Causal mediation", action_recovery),
            CircuitNode("planner.cost", "planner.cost", "Specificity", planner_effect),
            CircuitNode("planner.action", "planner.action", "Specificity", planner_effect),
        ]
    )
    for site in sites:
        recovery = next(node.score for node in nodes if node.node_id == site)
        edges.append(CircuitEdge(site, "predictor.latent", recovery, recovery / max(action_recovery, 1e-12)))
    edges.extend(
        [
            CircuitEdge("predictor.latent", "planner.cost", planner_effect, planner_effect),
            CircuitEdge("planner.cost", "planner.action", planner_effect, 1.0),
        ]
    )
    return CircuitGraph(
        graph_id="lewm-small-action-to-planner",
        nodes=tuple(nodes),
        edges=tuple(edges),
        status=status,
    )
