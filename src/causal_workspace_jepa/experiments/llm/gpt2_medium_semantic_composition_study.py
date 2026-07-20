"""Bounded semantic-direction and composition study on cached GPT-2 Medium."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.activation_store import estimate_storage
from causal_workspace_jepa.experiments.llm.gpt2_medium_mechanistic_study import (
    _evaluate_predictions,
    _select_train_logit_ids,
    _sha256,
    _validate_cpu_limits,
)
from causal_workspace_jepa.models.intervention_jepa import (
    RidgeInterventionPredictor,
    TinyMLPInterventionPredictor,
)


def run_gpt2_medium_semantic_composition_study(
    config_path: str | Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = time.perf_counter()
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/cpu_vps.yaml"))
    resource_report = require_free_disk(resource_profile)
    seed = int(config.get("seed", 11))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(int(config.get("max_cpu_threads", 4)), 4))
    model_name = str(config.get("model", "gpt2-medium"))
    cache_dir = str(config.get("cache_dir", ".cache/huggingface"))
    max_length = int(config.get("max_sequence_length", 24))
    layer = int(config.get("intervention_layer", 12))
    prompts = _evaluation_prompts(int(config.get("prompts", 6)))
    train_prompt_count = int(config.get("train_prompts", 4))
    calibration_prompts = _semantic_calibration_prompts()
    _validate_cpu_limits(max_length, [layer], prompts, train_prompt_count)

    epsilon = float(config.get("jacobian_epsilon", 0.5))
    large_magnitude = float(config.get("large_magnitude", 6.0))
    grid = _intervention_grid(epsilon, large_magnitude)
    example_count = len(prompts) * len(grid)
    if example_count != int(config.get("expected_examples", 72)):
        raise RuntimeError(f"configured semantic grid produced {example_count} examples")
    if len(grid) > int(config.get("max_intervention_batches", 12)):
        raise RuntimeError("BLOCKED_RESOURCE: semantic intervention batch limit exceeded")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=bool(config.get("local_files_only", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=bool(config.get("local_files_only", True)),
    )
    model.eval()
    model.config.use_cache = False

    calibration_batch = tokenizer(
        calibration_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    calibration_positions = calibration_batch["attention_mask"].sum(dim=1) - 1
    with torch.no_grad():
        calibration_run = model(
            **calibration_batch,
            output_hidden_states=True,
            use_cache=False,
        )
    calibration_source = _last_token(
        calibration_run.hidden_states[layer + 1],
        calibration_positions,
    )
    directions, direction_metrics = _construct_semantic_directions(calibration_source)
    del calibration_run, calibration_source

    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    positions = batch["attention_mask"].sum(dim=1) - 1
    with torch.no_grad():
        clean = model(**batch, output_hidden_states=True, use_cache=False)
    clean_source = _last_token(clean.hidden_states[layer + 1], positions).detach()
    clean_final = _last_token(clean.hidden_states[-1], positions).detach()
    clean_logits = _last_token(clean.logits, positions).detach()
    hidden_size = int(clean_source.shape[-1])
    hidden_ids = np.linspace(
        0,
        hidden_size - 1,
        num=int(config.get("hidden_outputs", 24)),
        dtype=np.int64,
    )
    logit_ids = _select_train_logit_ids(
        clean_logits[:train_prompt_count],
        count=int(config.get("logit_outputs", 24)),
    )
    storage = estimate_storage(
        examples=example_count,
        layers=2,
        positions=1,
        hidden_size=hidden_size,
        bytes_per_value=2,
    )
    target_bytes = example_count * (len(hidden_ids) + len(logit_ids)) * 2
    estimated_bytes = storage.estimated_bytes + target_bytes + directions.numel() * 2
    budget_bytes = int(float(config.get("activation_budget_mb", 16)) * 1024**2)
    if estimated_bytes > budget_bytes:
        raise RuntimeError(
            f"BLOCKED_RESOURCE: estimated activation data {estimated_bytes} exceeds {budget_bytes}"
        )

    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    prompt_ids: list[int] = []
    operation_ids: list[int] = []
    coefficients: list[np.ndarray] = []
    selected_logit_effects: list[float] = []
    top_token_changed: list[bool] = []
    for operation, coefficient in grid:
        if time.perf_counter() - started_at > float(config.get("max_runtime_seconds", 600)):
            raise RuntimeError("BLOCKED_RESOURCE: semantic study exceeded CPU runtime guard")
        edit = coefficient[0] * directions[0] + coefficient[1] * directions[1]
        with torch.no_grad():
            intervened = _forward_batch_with_residual_edit(
                model=model,
                batch=batch,
                layer=layer,
                positions=positions,
                edit=edit,
            )
        final = _last_token(intervened.hidden_states[-1], positions).detach()
        logits = _last_token(intervened.logits, positions).detach()
        hidden_delta = final[:, hidden_ids] - clean_final[:, hidden_ids]
        logit_delta = logits[:, logit_ids] - clean_logits[:, logit_ids]
        combined = torch.cat([hidden_delta, logit_delta], dim=1)
        contexts.extend(clean_source.cpu().numpy().astype(np.float32))
        targets.extend(combined.cpu().numpy().astype(np.float32))
        prompt_ids.extend(range(len(prompts)))
        operation_ids.extend([int(operation == "composition")] * len(prompts))
        coefficients.extend([coefficient.copy() for _ in prompts])
        selected_logit_effects.extend(
            torch.mean(torch.abs(logit_delta), dim=1).cpu().tolist()
        )
        top_token_changed.extend(
            (torch.argmax(logits, dim=1) != torch.argmax(clean_logits, dim=1)).cpu().tolist()
        )
        del intervened, final, logits, hidden_delta, logit_delta, combined

    context_array = np.asarray(contexts, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    prompt_array = np.asarray(prompt_ids, dtype=np.int16)
    operation_array = np.asarray(operation_ids, dtype=np.int8)
    coefficient_array = np.asarray(coefficients, dtype=np.float32)
    projection_rng = np.random.default_rng(seed + 91)
    projection = projection_rng.normal(
        0.0,
        1.0 / np.sqrt(hidden_size),
        size=(hidden_size, int(config.get("context_projection_dim", 24))),
    ).astype(np.float32)
    projected_context = context_array @ projection
    train_mask, primary_mask, seen_composition_mask = _build_semantic_split_masks(
        prompt_array,
        operation_array,
        train_prompt_count=train_prompt_count,
    )
    if not train_mask.any() or not primary_mask.any() or not seen_composition_mask.any():
        raise RuntimeError("configured semantic split produced an empty partition")

    linear = RidgeInterventionPredictor.fit(
        projected_context[train_mask],
        coefficient_array[train_mask],
        target_array[train_mask],
        mode="linear",
        ridge=float(config.get("ridge", 3e-3)),
    )
    bilinear = RidgeInterventionPredictor.fit(
        projected_context[train_mask],
        coefficient_array[train_mask],
        target_array[train_mask],
        mode="bilinear",
        ridge=float(config.get("ridge", 3e-3)),
    )
    mlp = TinyMLPInterventionPredictor.fit(
        projected_context[train_mask],
        coefficient_array[train_mask],
        target_array[train_mask],
        hidden_dim=int(config.get("mlp_hidden_dim", 48)),
        steps=int(config.get("mlp_steps", 300)),
        learning_rate=float(config.get("mlp_learning_rate", 0.01)),
        seed=seed,
    )
    train_mean = target_array[train_mask].mean(axis=0)
    split_metrics = {}
    for split_name, mask in (
        ("heldout_prompt_compositions", primary_mask),
        ("seen_prompt_compositions", seen_composition_mask),
    ):
        predictions = {
            "no_change": np.zeros_like(target_array[mask]),
            "mean_effect": np.broadcast_to(train_mean, target_array[mask].shape),
            "linear_regression": linear.predict(
                projected_context[mask], coefficient_array[mask]
            ),
            "bilinear_intervention_jepa": bilinear.predict(
                projected_context[mask], coefficient_array[mask]
            ),
            "mlp": mlp.predict(projected_context[mask], coefficient_array[mask]),
            "prompt_local_additive_jacobian": _additive_jacobian_predictions(
                target_array,
                prompt_array,
                operation_array,
                coefficient_array,
                mask,
                epsilon=epsilon,
                local=True,
                train_prompt_count=train_prompt_count,
            ),
            "corpus_averaged_additive_jacobian": _additive_jacobian_predictions(
                target_array,
                prompt_array,
                operation_array,
                coefficient_array,
                mask,
                epsilon=epsilon,
                local=False,
                train_prompt_count=train_prompt_count,
            ),
            "prompt_local_large_additive": _large_additive_predictions(
                target_array,
                prompt_array,
                operation_array,
                coefficient_array,
                mask,
            ),
        }
        split_metrics[split_name] = _evaluate_predictions(
            predictions,
            target_array[mask],
            hidden_output_count=len(hidden_ids),
        )
        split_metrics[split_name]["composition_nonlinearity"] = _composition_nonlinearity(
            predictions["prompt_local_large_additive"],
            target_array[mask],
        )

    primary_scores = split_metrics["heldout_prompt_compositions"]["models"]
    learned_names = ("bilinear_intervention_jepa", "mlp")
    best_learned = min(learned_names, key=lambda name: primary_scores[name]["mse"])
    nonlinearity = split_metrics["heldout_prompt_compositions"]["composition_nonlinearity"]
    minimum_nonlinearity = float(config.get("min_nonlinearity_fraction", 0.05))
    nonlinear_advantage = bool(
        nonlinearity["interaction_fraction_of_effect"] >= minimum_nonlinearity
        and primary_scores["mlp"]["mse"]
        < primary_scores["prompt_local_additive_jacobian"]["mse"]
    )
    causal_compression = bool(
        primary_scores["bilinear_intervention_jepa"]["mse"]
        < min(primary_scores["no_change"]["mse"], primary_scores["linear_regression"]["mse"])
    )
    compositional_generalization = bool(
        primary_scores[best_learned]["mse"]
        < min(
            primary_scores["no_change"]["mse"],
            primary_scores["corpus_averaged_additive_jacobian"]["mse"],
        )
        and primary_scores[best_learned]["effect_correlation"]
        >= float(config.get("min_effect_correlation", 0.5))
    )

    if time.perf_counter() - started_at > float(config.get("max_runtime_seconds", 600)):
        raise RuntimeError("BLOCKED_RESOURCE: semantic study exceeded CPU runtime guard")
    dataset_path = Path(
        str(config.get("dataset_output", "data/activations/gpt2_medium_semantic_v3.npz"))
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        contexts=context_array.astype(np.float16),
        targets=target_array.astype(np.float16),
        prompt_ids=prompt_array,
        operation_ids=operation_array,
        coefficients=coefficient_array,
        directions=directions.cpu().numpy().astype(np.float16),
        hidden_output_ids=hidden_ids,
        logit_output_ids=logit_ids,
    )
    dataset_checksum = _sha256(dataset_path)
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path}",
        resource_profile=resource_profile,
        seed=seed,
    )
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "LLM-GPT2-003")),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Generalization",
        "model": model_name,
        "seed": seed,
        "intervention_layer": layer,
        "intervention_site": f"transformer.h.{layer}.resid_post",
        "prompts": len(prompts),
        "train_prompts": train_prompt_count,
        "calibration_prompts": len(calibration_prompts),
        "calibration_prompt_sha256": _text_sha256(calibration_prompts),
        "evaluation_prompt_sha256": _text_sha256(prompts),
        "direction_construction": direction_metrics,
        "jacobian_epsilon": epsilon,
        "large_magnitude": large_magnitude,
        "examples": example_count,
        "train_single_examples": int(train_mask.sum()),
        "heldout_prompt_composition_examples": int(primary_mask.sum()),
        "seen_prompt_composition_examples": int(seen_composition_mask.sum()),
        "activation_storage_estimate_bytes": estimated_bytes,
        "activation_budget_bytes": budget_bytes,
        "dataset_bytes": dataset_path.stat().st_size,
        "dataset_sha256": dataset_checksum,
        "free_disk_gb_before_model_run": round(resource_report.free_gb, 3),
        "runtime_seconds": float(time.perf_counter() - started_at),
        "direct_selected_logit_mean_abs_delta": float(np.mean(selected_logit_effects)),
        "direct_top_token_change_rate": float(np.mean(top_token_changed)),
        "composition_top_token_change_rate": float(
            np.mean(np.asarray(top_token_changed)[operation_array == 1])
        ),
        "splits": split_metrics,
        "best_heldout_composition_learned_model": best_learned,
        "hypotheses": [
            {
                "id": "H-LLM-01",
                "claim": "The MLP predictor beats a prompt-local additive Jacobian.",
                "evidence_level": "Generalization",
                "supported": nonlinear_advantage,
            },
            {
                "id": "H-LLM-02",
                "claim": "Bilinear causal compression beats no-change and linear regression.",
                "evidence_level": "Generalization",
                "supported": causal_compression,
            },
            {
                "id": "H-LLM-03",
                "claim": "A singles-only predictor generalizes to held-out prompt compositions.",
                "evidence_level": "Generalization",
                "supported": compositional_generalization,
            },
        ],
        "limitations": [
            "GPT-2 Medium is not Qwen.",
            "Construction labels for contrast directions are not validated feature semantics.",
            "The study uses one layer, six prompts, two directions, selected outputs, and one seed.",
            "No result establishes J-space, a workspace, or behavior-level mechanism reuse.",
        ],
    }
    metrics_path = Path(
        str(
            config.get(
                "output_metrics",
                "artifacts/metrics/gpt2_medium_semantic_composition_study.json",
            )
        )
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = Path(
        str(config.get("output_manifest", "data/manifests/gpt2_medium_semantic_v3.json"))
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "canonical_name": "gpt2-medium-semantic-composition-v3",
                "status": "GENERATED_LOCAL",
                "source": "locally cached Hugging Face gpt2-medium",
                "version": f"LLM-GPT2-003 seed {seed}",
                "license": "MIT model license; repository-authored prompts",
                "path": str(dataset_path),
                "sha256": dataset_checksum,
                "bytes": dataset_path.stat().st_size,
                "compressed_bytes": dataset_path.stat().st_size,
                "modalities": [
                    "residual_state",
                    "contrast_direction",
                    "intervention_composition",
                    "hidden_delta",
                    "logit_delta",
                ],
                "examples": example_count,
                "dtype": "float16",
                "shapes": {
                    "context": [example_count, hidden_size],
                    "target": [example_count, len(hidden_ids) + len(logit_ids)],
                    "directions": [2, hidden_size],
                },
                "selected_layers": [layer],
                "selected_positions": "last_non_padding_token",
                "prompt_split": {
                    "train": train_prompt_count,
                    "test": len(prompts) - train_prompt_count,
                    "calibration_disjoint": len(calibration_prompts),
                },
                "leakage_controls": [
                    "disjoint_direction_calibration_prompts",
                    "heldout_evaluation_prompts",
                    "composition_excluded_from_training",
                ],
                "download_command": "none; local_files_only=true",
                "generation_command": (
                    "python scripts/run_experiment.py --config "
                    "configs/experiments/gpt2_medium_semantic_composition_study.yaml"
                ),
                "resource_modes_allowed": ["cpu_vps_explicit_override", "gpu_12gb"],
                "purpose": "direct GPT-2 composition nonlinearity and meta-model fidelity",
                "config": str(config_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_provenance(
        metrics_path.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": str(metrics_path), "dataset_sha256": dataset_checksum},
    )
    return metrics


def _intervention_grid(
    epsilon: float,
    large_magnitude: float,
) -> list[tuple[str, np.ndarray]]:
    if not 0.0 < epsilon < large_magnitude:
        raise ValueError("epsilon must be positive and smaller than large_magnitude")
    grid = []
    for direction in range(2):
        for magnitude in (-epsilon, epsilon, -large_magnitude, large_magnitude):
            coefficient = np.zeros(2, dtype=np.float32)
            coefficient[direction] = magnitude
            grid.append(("single", coefficient))
    for first in (-large_magnitude, large_magnitude):
        for second in (-large_magnitude, large_magnitude):
            grid.append(("composition", np.asarray([first, second], dtype=np.float32)))
    return grid


def _build_semantic_split_masks(
    prompt_ids: np.ndarray,
    operation_ids: np.ndarray,
    *,
    train_prompt_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    single = operation_ids == 0
    composition = operation_ids == 1
    train = (prompt_ids < train_prompt_count) & single
    primary = (prompt_ids >= train_prompt_count) & composition
    seen_composition = (prompt_ids < train_prompt_count) & composition
    return train, primary, seen_composition


def _additive_jacobian_predictions(
    targets: np.ndarray,
    prompt_ids: np.ndarray,
    operation_ids: np.ndarray,
    coefficients: np.ndarray,
    test_mask: np.ndarray,
    *,
    epsilon: float,
    local: bool,
    train_prompt_count: int,
) -> np.ndarray:
    predictions = []
    for index in np.flatnonzero(test_mask):
        prompt_pool = [int(prompt_ids[index])] if local else list(range(train_prompt_count))
        slopes = []
        for direction in range(2):
            prompt_slopes = [
                _finite_difference_slope(
                    targets,
                    prompt_ids,
                    operation_ids,
                    coefficients,
                    prompt=prompt,
                    direction=direction,
                    epsilon=epsilon,
                )
                for prompt in prompt_pool
            ]
            slopes.append(np.mean(prompt_slopes, axis=0))
        predictions.append(
            slopes[0] * coefficients[index, 0] + slopes[1] * coefficients[index, 1]
        )
    return np.asarray(predictions, dtype=np.float32)


def _finite_difference_slope(
    targets: np.ndarray,
    prompt_ids: np.ndarray,
    operation_ids: np.ndarray,
    coefficients: np.ndarray,
    *,
    prompt: int,
    direction: int,
    epsilon: float,
) -> np.ndarray:
    other = 1 - direction
    common = (
        (prompt_ids == prompt)
        & (operation_ids == 0)
        & np.isclose(coefficients[:, other], 0.0)
    )
    plus = np.flatnonzero(common & np.isclose(coefficients[:, direction], epsilon))
    minus = np.flatnonzero(common & np.isclose(coefficients[:, direction], -epsilon))
    if len(plus) != 1 or len(minus) != 1:
        raise RuntimeError("semantic finite-difference pair is missing or duplicated")
    return (targets[plus[0]] - targets[minus[0]]) / (2.0 * epsilon)


def _large_additive_predictions(
    targets: np.ndarray,
    prompt_ids: np.ndarray,
    operation_ids: np.ndarray,
    coefficients: np.ndarray,
    test_mask: np.ndarray,
) -> np.ndarray:
    predictions = []
    for index in np.flatnonzero(test_mask):
        parts = []
        for direction in range(2):
            other = 1 - direction
            match = (
                (prompt_ids == prompt_ids[index])
                & (operation_ids == 0)
                & np.isclose(coefficients[:, direction], coefficients[index, direction])
                & np.isclose(coefficients[:, other], 0.0)
            )
            ids = np.flatnonzero(match)
            if len(ids) != 1:
                raise RuntimeError("large single-effect row is missing or duplicated")
            parts.append(targets[ids[0]])
        predictions.append(parts[0] + parts[1])
    return np.asarray(predictions, dtype=np.float32)


def _composition_nonlinearity(
    additive_prediction: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float]:
    interaction_mse = float(np.mean((observed - additive_prediction) ** 2))
    effect_power = float(np.mean(observed**2))
    return {
        "interaction_mse": interaction_mse,
        "interaction_rms": float(np.sqrt(interaction_mse)),
        "effect_power": effect_power,
        "interaction_fraction_of_effect": float(interaction_mse / max(effect_power, 1e-12)),
    }


def _construct_semantic_directions(activations: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    if activations.shape[0] != 8:
        raise ValueError("semantic direction construction requires eight calibration activations")
    sentiment_raw = activations[:2].mean(dim=0) - activations[2:4].mean(dim=0)
    domain_raw = activations[4:6].mean(dim=0) - activations[6:8].mean(dim=0)
    sentiment_norm = torch.linalg.vector_norm(sentiment_raw)
    domain_raw_norm = torch.linalg.vector_norm(domain_raw)
    if float(sentiment_norm) <= 1e-8 or float(domain_raw_norm) <= 1e-8:
        raise RuntimeError("semantic calibration produced a degenerate contrast")
    raw_cosine = torch.dot(sentiment_raw, domain_raw) / (
        sentiment_norm * domain_raw_norm
    )
    sentiment = sentiment_raw / sentiment_norm
    domain = domain_raw - torch.dot(domain_raw, sentiment) * sentiment
    domain_norm = torch.linalg.vector_norm(domain)
    if float(domain_norm) <= 1e-8:
        raise RuntimeError("semantic calibration contrasts became collinear")
    domain = domain / domain_norm
    directions = torch.stack([sentiment, domain])
    return directions, {
        "labels_are_construction_only": True,
        "directions": ["sentiment_contrast", "geography_vs_biology_contrast"],
        "sentiment_raw_norm": float(sentiment_norm),
        "domain_raw_norm": float(domain_raw_norm),
        "raw_cosine": float(raw_cosine),
        "orthogonalized_cosine": float(torch.dot(sentiment, domain)),
    }


def _forward_batch_with_residual_edit(
    *,
    model: Any,
    batch: dict[str, Any],
    layer: int,
    positions: Any,
    edit: Any,
) -> Any:
    block = model.transformer.h[layer]

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
        indexes = positions.new_tensor(list(range(hidden.shape[0])))
        hidden[indexes, positions, :] += edit.to(hidden.device)
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    handle = block.register_forward_hook(hook)
    try:
        return model(**batch, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()


def _last_token(tensor: Any, positions: Any) -> Any:
    import torch

    indexes = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[indexes, positions.to(tensor.device)]


def _text_sha256(prompts: list[str]) -> str:
    return hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()


def _semantic_calibration_prompts() -> list[str]:
    return [
        "The reviewer called the film wonderful and felt delighted",
        "The customer said the service was excellent and left happy",
        "The reviewer called the film awful and felt disappointed",
        "The customer said the service was terrible and left unhappy",
        "The atlas marks borders capitals rivers and countries",
        "The traveler studied maps of cities oceans and nations",
        "The textbook explains cells proteins genes and organisms",
        "The scientist studied tissues enzymes DNA and species",
    ]


def _evaluation_prompts(count: int) -> list[str]:
    prompts = [
        "The new restaurant received a careful review from the local newspaper",
        "A student compared two explanations before writing the final answer",
        "The committee discussed the city proposal during the evening meeting",
        "An engineer inspected the device and recorded the unexpected result",
        "The museum guide described the ancient exhibit to a crowded room",
        "A journalist summarized the scientific report for a general audience",
    ]
    if count != len(prompts):
        raise ValueError("LLM-GPT2-003 requires exactly six fixed evaluation prompts")
    return prompts
