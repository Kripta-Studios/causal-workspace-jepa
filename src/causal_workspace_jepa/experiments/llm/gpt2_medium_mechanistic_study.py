"""Held-out GPT-2 Medium intervention-effect prediction study."""

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
from causal_workspace_jepa.models.intervention_jepa import (
    RidgeInterventionPredictor,
    TinyMLPInterventionPredictor,
    effect_correlation,
)


def run_gpt2_medium_mechanistic_study(config_path: str | Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = time.perf_counter()
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/cpu_vps.yaml"))
    resource_report = require_free_disk(resource_profile)
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_name = str(config.get("model", "gpt2-medium"))
    cache_dir = str(config.get("cache_dir", ".cache/huggingface"))
    max_length = int(config.get("max_sequence_length", 24))
    layers = [int(value) for value in config.get("selected_layers", [6, 12, 18])]
    train_layers = [int(value) for value in config.get("train_layers", [6, 12])]
    heldout_layer = int(config.get("heldout_layer", 18))
    magnitudes = [
        float(value)
        for value in config.get("magnitudes", [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])
    ]
    jacobian_epsilon = float(config.get("jacobian_epsilon", 0.25))
    direction_count = int(config.get("intervention_directions", 2))
    prompts = _study_prompts(int(config.get("prompts", 8)))
    train_prompt_count = int(config.get("train_prompts", 6))
    _validate_cpu_limits(max_length, layers, prompts, train_prompt_count)

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
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    positions = batch["attention_mask"].sum(dim=1) - 1
    batch_indices = torch.arange(len(prompts))
    with torch.no_grad():
        clean = model(**batch, output_hidden_states=True)
    hidden_size = int(clean.hidden_states[-1].shape[-1])
    clean_final = clean.hidden_states[-1][batch_indices, positions].detach()
    clean_logits = clean.logits[batch_indices, positions].detach()
    hidden_ids = np.linspace(
        0,
        hidden_size - 1,
        num=int(config.get("hidden_outputs", 32)),
        dtype=np.int64,
    )
    logit_ids = _select_train_logit_ids(
        clean_logits[:train_prompt_count],
        count=int(config.get("logit_outputs", 32)),
    )
    direction_ids = np.linspace(0, hidden_size - 1, num=direction_count, dtype=np.int64)
    example_count = len(prompts) * len(layers) * direction_count * len(magnitudes)
    storage = estimate_storage(
        examples=example_count,
        layers=2,
        positions=1,
        hidden_size=hidden_size,
        bytes_per_value=2,
    )
    target_bytes = example_count * (len(hidden_ids) + len(logit_ids)) * 2
    estimated_bytes = storage.estimated_bytes + target_bytes
    budget_bytes = int(float(config.get("activation_budget_mb", 64)) * 1024**2)
    if estimated_bytes > budget_bytes:
        raise RuntimeError(
            f"BLOCKED_RESOURCE: estimated activation data {estimated_bytes} exceeds {budget_bytes}"
        )

    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    prompt_ids: list[int] = []
    layer_ids: list[int] = []
    direction_indexes: list[int] = []
    example_magnitudes: list[float] = []
    top_token_changed: list[bool] = []
    selected_logit_effects: list[float] = []
    for layer in layers:
        clean_source = clean.hidden_states[layer + 1][batch_indices, positions].detach()
        for direction_index, feature_id in enumerate(direction_ids):
            direction = torch.zeros(hidden_size, dtype=clean_source.dtype)
            direction[int(feature_id)] = 1.0
            for magnitude in magnitudes:
                with torch.no_grad():
                    intervened = _forward_batch_with_residual_steer(
                        model=model,
                        batch=batch,
                        layer=layer,
                        positions=positions,
                        direction=direction,
                        magnitude=magnitude,
                    )
                final = intervened.hidden_states[-1][batch_indices, positions].detach()
                logits = intervened.logits[batch_indices, positions].detach()
                hidden_delta = final[:, hidden_ids] - clean_final[:, hidden_ids]
                logit_delta = logits[:, logit_ids] - clean_logits[:, logit_ids]
                combined = torch.cat([hidden_delta, logit_delta], dim=1)
                contexts.extend(clean_source.cpu().numpy().astype(np.float32))
                targets.extend(combined.cpu().numpy().astype(np.float32))
                prompt_ids.extend(range(len(prompts)))
                layer_ids.extend([layer] * len(prompts))
                direction_indexes.extend([direction_index] * len(prompts))
                example_magnitudes.extend([magnitude] * len(prompts))
                changed = torch.argmax(logits, dim=1) != torch.argmax(clean_logits, dim=1)
                top_token_changed.extend(changed.cpu().tolist())
                selected_logit_effects.extend(
                    torch.mean(torch.abs(logit_delta), dim=1).cpu().tolist()
                )
                del intervened, final, logits, hidden_delta, logit_delta, combined

    context_array = np.asarray(contexts, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    prompt_array = np.asarray(prompt_ids, dtype=np.int16)
    layer_array = np.asarray(layer_ids, dtype=np.int16)
    direction_array = np.asarray(direction_indexes, dtype=np.int16)
    magnitude_array = np.asarray(example_magnitudes, dtype=np.float32)
    intervention_array = np.stack(
        [
            _intervention_features(layer, direction, magnitude, layers, direction_count)
            for layer, direction, magnitude in zip(layer_array, direction_array, magnitude_array)
        ]
    )
    projection_rng = np.random.default_rng(seed + 81)
    projection = projection_rng.normal(
        0.0,
        1.0 / np.sqrt(hidden_size),
        size=(hidden_size, int(config.get("context_projection_dim", 32))),
    ).astype(np.float32)
    projected_context = context_array @ projection

    train_mask, prompt_test_mask, layer_test_mask = _build_split_masks(
        prompt_array,
        layer_array,
        magnitude_array,
        train_prompt_count=train_prompt_count,
        train_layers=train_layers,
        heldout_layer=heldout_layer,
        max_train_magnitude=float(config.get("max_train_magnitude", 0.5)),
        test_magnitude=float(config.get("test_magnitude", 1.0)),
    )
    if not train_mask.any() or not prompt_test_mask.any() or not layer_test_mask.any():
        raise RuntimeError("configured GPT-2 split produced an empty partition")

    linear = RidgeInterventionPredictor.fit(
        projected_context[train_mask],
        intervention_array[train_mask],
        target_array[train_mask],
        mode="linear",
        ridge=float(config.get("ridge", 1e-3)),
    )
    bilinear = RidgeInterventionPredictor.fit(
        projected_context[train_mask],
        intervention_array[train_mask],
        target_array[train_mask],
        mode="bilinear",
        ridge=float(config.get("ridge", 1e-3)),
    )
    mlp = TinyMLPInterventionPredictor.fit(
        projected_context[train_mask],
        intervention_array[train_mask],
        target_array[train_mask],
        hidden_dim=int(config.get("mlp_hidden_dim", 64)),
        steps=int(config.get("mlp_steps", 400)),
        learning_rate=float(config.get("mlp_learning_rate", 0.01)),
        seed=seed,
    )
    sparse_context = _topk_sparse(projected_context, int(config.get("sparse_context_features", 8)))
    sparse_linear = RidgeInterventionPredictor.fit(
        sparse_context[train_mask],
        intervention_array[train_mask],
        target_array[train_mask],
        mode="linear",
        ridge=float(config.get("ridge", 1e-3)),
    )
    train_target_mean = target_array[train_mask].mean(axis=0)

    split_metrics = {}
    for split_name, mask in (
        ("heldout_prompt_and_magnitude", prompt_test_mask),
        ("heldout_layer_prompt_and_magnitude", layer_test_mask),
    ):
        predictions = {
            "no_change": np.zeros_like(target_array[mask]),
            "mean_effect": np.broadcast_to(train_target_mean, target_array[mask].shape),
            "linear_regression": linear.predict(projected_context[mask], intervention_array[mask]),
            "bilinear_intervention_jepa": bilinear.predict(
                projected_context[mask], intervention_array[mask]
            ),
            "mlp": mlp.predict(projected_context[mask], intervention_array[mask]),
            "sparse_context_linear": sparse_linear.predict(
                sparse_context[mask], intervention_array[mask]
            ),
            "nearest_neighbor": _nearest_neighbor_prediction(
                projected_context[train_mask],
                intervention_array[train_mask],
                target_array[train_mask],
                projected_context[mask],
                intervention_array[mask],
            ),
            "local_finite_difference_jacobian": _jacobian_predictions(
                target_array,
                prompt_array,
                layer_array,
                direction_array,
                magnitude_array,
                mask,
                epsilon=jacobian_epsilon,
                local=True,
                train_prompt_count=train_prompt_count,
                train_layers=train_layers,
            ),
            "corpus_averaged_jacobian": _jacobian_predictions(
                target_array,
                prompt_array,
                layer_array,
                direction_array,
                magnitude_array,
                mask,
                epsilon=jacobian_epsilon,
                local=False,
                train_prompt_count=train_prompt_count,
                train_layers=train_layers,
            ),
        }
        split_metrics[split_name] = _evaluate_predictions(
            predictions,
            target_array[mask],
            hidden_output_count=len(hidden_ids),
        )

    main_scores = split_metrics["heldout_prompt_and_magnitude"]["models"]
    best_name = min(main_scores, key=lambda name: main_scores[name]["mse"])
    nonlinear_best = min(
        main_scores["bilinear_intervention_jepa"]["mse"],
        main_scores["mlp"]["mse"],
    )
    nonlinear_advantage = bool(
        nonlinear_best < main_scores["local_finite_difference_jacobian"]["mse"]
    )
    causal_compression = bool(
        main_scores["bilinear_intervention_jepa"]["mse"]
        < min(main_scores["no_change"]["mse"], main_scores["linear_regression"]["mse"])
    )
    direct_verification = bool(main_scores[best_name]["effect_correlation"] > 0.5)

    dataset_path = Path(str(config.get("dataset_output", "data/activations/gpt2_medium_v2.npz")))
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        contexts=context_array.astype(np.float16),
        targets=target_array.astype(np.float16),
        prompt_ids=prompt_array,
        layer_ids=layer_array,
        direction_ids=direction_array,
        magnitudes=magnitude_array,
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
        "experiment_id": config.get("id", "LLM-GPT2-002"),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Causal mediation",
        "model": model_name,
        "seed": seed,
        "prompts": len(prompts),
        "train_prompts": train_prompt_count,
        "selected_layers": layers,
        "train_layers": train_layers,
        "heldout_layer": heldout_layer,
        "magnitudes": magnitudes,
        "directions": direction_count,
        "max_sequence_length": max_length,
        "examples": example_count,
        "train_examples": int(train_mask.sum()),
        "heldout_prompt_examples": int(prompt_test_mask.sum()),
        "heldout_layer_examples": int(layer_test_mask.sum()),
        "activation_storage_estimate_bytes": estimated_bytes,
        "activation_budget_bytes": budget_bytes,
        "dataset_bytes": dataset_path.stat().st_size,
        "dataset_sha256": dataset_checksum,
        "free_disk_gb_before_model_run": round(resource_report.free_gb, 3),
        "runtime_seconds": float(time.perf_counter() - started_at),
        "direct_selected_logit_mean_abs_delta": float(np.mean(selected_logit_effects)),
        "direct_top_token_change_rate": float(np.mean(top_token_changed)),
        "splits": split_metrics,
        "best_heldout_prompt_model": best_name,
        "hypotheses": [
            {
                "id": "H-LLM-01",
                "claim": "A nonlinear intervention predictor beats a prompt-local Jacobian.",
                "evidence_level": "Causal mediation",
                "supported": nonlinear_advantage,
            },
            {
                "id": "H-LLM-02",
                "claim": "The bilinear meta-state beats no-change and linear regression.",
                "evidence_level": "Causal mediation",
                "supported": causal_compression,
            },
            {
                "id": "H-LLM-06",
                "claim": "The best predicted effects correlate with direct execution above 0.5.",
                "evidence_level": "Causal mediation",
                "supported": direct_verification,
            },
        ],
        "limitations": [
            "GPT-2 Medium is not Qwen.",
            "Coordinate steering is not a discovered semantic feature or J-space direction.",
            (
                "The study has one seed, eight local prompts, selected positions, and selected "
                "outputs."
            ),
        ],
    }
    metrics_path = Path(
        str(config.get("output_metrics", "artifacts/metrics/gpt2_medium_mechanistic_study.json"))
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = Path(
        str(config.get("output_manifest", "data/manifests/gpt2_medium_interventions.json"))
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "canonical_name": "gpt2-medium-residual-interventions-v2",
                "status": "GENERATED_LOCAL",
                "source": "locally cached Hugging Face gpt2-medium",
                "version": "LLM-GPT2-002 seed 0",
                "license": "MIT model license; repository-authored prompts",
                "path": str(dataset_path),
                "sha256": dataset_checksum,
                "bytes": dataset_path.stat().st_size,
                "compressed_bytes": dataset_path.stat().st_size,
                "modalities": ["residual_state", "intervention", "hidden_delta", "logit_delta"],
                "examples": example_count,
                "dtype": "float16",
                "shapes": {
                    "context": [example_count, hidden_size],
                    "target": [example_count, len(hidden_ids) + len(logit_ids)],
                },
                "selected_layers": layers,
                "selected_positions": "last_non_padding_token",
                "prompt_split": {
                    "train": train_prompt_count,
                    "test": len(prompts) - train_prompt_count,
                },
                "leakage_controls": ["prompt", "magnitude", "layer"],
                "download_command": "none; local_files_only=true",
                "generation_command": (
                    "python scripts/run_experiment.py --config "
                    "configs/experiments/gpt2_medium_mechanistic_study.yaml"
                ),
                "resource_modes_allowed": ["cpu_vps_explicit_override", "gpu_12gb"],
                "purpose": "held-out prediction of direct GPT-2 residual intervention effects",
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


def _validate_cpu_limits(
    max_length: int,
    layers: list[int],
    prompts: list[str],
    train_prompt_count: int,
) -> None:
    if max_length > 64:
        raise RuntimeError("BLOCKED_RESOURCE: cpu_vps GPT-2 sequence length must be <= 64")
    if len(layers) > 6:
        raise RuntimeError("BLOCKED_RESOURCE: cpu_vps GPT-2 capture is limited to six layers")
    if len(prompts) > 200:
        raise RuntimeError("BLOCKED_RESOURCE: cpu_vps GPT-2 smoke is limited to 200 prompts")
    if train_prompt_count <= 0 or train_prompt_count >= len(prompts):
        raise ValueError("train_prompts must leave at least one held-out prompt")


def _forward_batch_with_residual_steer(
    *,
    model: Any,
    batch: dict[str, Any],
    layer: int,
    positions: Any,
    direction: Any,
    magnitude: float,
) -> Any:
    block = model.transformer.h[layer]

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
        indexes = positions.new_tensor(list(range(hidden.shape[0])))
        hidden[indexes, positions, :] += magnitude * direction.to(hidden.device)
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    handle = block.register_forward_hook(hook)
    try:
        return model(**batch, output_hidden_states=True)
    finally:
        handle.remove()


def _select_train_logit_ids(logits: Any, count: int) -> np.ndarray:
    ranked = logits.topk(min(count, logits.shape[-1]), dim=1).indices.cpu().numpy()
    candidates: list[int] = []
    seen: set[int] = set()
    for rank in range(ranked.shape[1]):
        for prompt in range(ranked.shape[0]):
            token_id = int(ranked[prompt, rank])
            if token_id not in seen:
                seen.add(token_id)
                candidates.append(token_id)
            if len(candidates) == count:
                return np.asarray(candidates, dtype=np.int64)
    for token_id in range(logits.shape[-1]):
        if token_id not in seen:
            candidates.append(token_id)
        if len(candidates) == count:
            break
    return np.asarray(candidates, dtype=np.int64)


def _build_split_masks(
    prompts: np.ndarray,
    layers: np.ndarray,
    magnitudes: np.ndarray,
    *,
    train_prompt_count: int,
    train_layers: list[int],
    heldout_layer: int,
    max_train_magnitude: float,
    test_magnitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    small = np.abs(magnitudes) <= max_train_magnitude + 1e-8
    large = np.isclose(np.abs(magnitudes), test_magnitude)
    train = (prompts < train_prompt_count) & np.isin(layers, train_layers) & small
    prompt_test = (prompts >= train_prompt_count) & np.isin(layers, train_layers) & large
    layer_test = (prompts >= train_prompt_count) & (layers == heldout_layer) & large
    return train, prompt_test, layer_test


def _intervention_features(
    layer: int,
    direction: int,
    magnitude: float,
    layers: list[int],
    direction_count: int,
) -> np.ndarray:
    result = np.zeros(len(layers) + direction_count + 1, dtype=np.float32)
    result[layers.index(int(layer))] = 1.0
    result[len(layers) + int(direction)] = 1.0
    result[-1] = float(magnitude)
    return result


def _jacobian_predictions(
    targets: np.ndarray,
    prompts: np.ndarray,
    layers: np.ndarray,
    directions: np.ndarray,
    magnitudes: np.ndarray,
    test_mask: np.ndarray,
    *,
    epsilon: float,
    local: bool,
    train_prompt_count: int,
    train_layers: list[int],
) -> np.ndarray:
    predictions = []
    test_indices = np.flatnonzero(test_mask)
    for index in test_indices:
        if local:
            slopes = [
                _finite_difference_slope(
                    targets,
                    prompts,
                    layers,
                    directions,
                    magnitudes,
                    prompt=int(prompts[index]),
                    layer=int(layers[index]),
                    direction=int(directions[index]),
                    epsilon=epsilon,
                )
            ]
        else:
            candidate_layers = (
                [int(layers[index])] if int(layers[index]) in train_layers else train_layers
            )
            slopes = [
                _finite_difference_slope(
                    targets,
                    prompts,
                    layers,
                    directions,
                    magnitudes,
                    prompt=prompt,
                    layer=layer,
                    direction=int(directions[index]),
                    epsilon=epsilon,
                )
                for prompt in range(train_prompt_count)
                for layer in candidate_layers
            ]
        predictions.append(np.mean(slopes, axis=0) * magnitudes[index])
    return np.asarray(predictions, dtype=np.float32)


def _finite_difference_slope(
    targets: np.ndarray,
    prompts: np.ndarray,
    layers: np.ndarray,
    directions: np.ndarray,
    magnitudes: np.ndarray,
    *,
    prompt: int,
    layer: int,
    direction: int,
    epsilon: float,
) -> np.ndarray:
    common = (prompts == prompt) & (layers == layer) & (directions == direction)
    plus = np.flatnonzero(common & np.isclose(magnitudes, epsilon))
    minus = np.flatnonzero(common & np.isclose(magnitudes, -epsilon))
    if len(plus) != 1 or len(minus) != 1:
        raise RuntimeError("finite-difference pair is missing or duplicated")
    return (targets[plus[0]] - targets[minus[0]]) / (2.0 * epsilon)


def _nearest_neighbor_prediction(
    train_context: np.ndarray,
    train_interventions: np.ndarray,
    train_targets: np.ndarray,
    test_context: np.ndarray,
    test_interventions: np.ndarray,
) -> np.ndarray:
    train = np.concatenate([train_context, train_interventions], axis=1)
    test = np.concatenate([test_context, test_interventions], axis=1)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    distances = np.sum(
        (((test[:, None, :] - mean) / scale) - ((train[None, :, :] - mean) / scale)) ** 2,
        axis=2,
    )
    return train_targets[np.argmin(distances, axis=1)]


def _topk_sparse(context: np.ndarray, features: int) -> np.ndarray:
    if features >= context.shape[1]:
        return context.copy()
    result = np.zeros_like(context)
    ids = np.argpartition(np.abs(context), -features, axis=1)[:, -features:]
    rows = np.arange(len(context))[:, None]
    result[rows, ids] = context[rows, ids]
    return result


def _evaluate_predictions(
    predictions: dict[str, np.ndarray],
    observed: np.ndarray,
    *,
    hidden_output_count: int,
) -> dict[str, object]:
    model_metrics = {}
    for name, predicted in predictions.items():
        model_metrics[name] = {
            "evidence_level": "Causal mediation",
            "mse": float(np.mean((predicted - observed) ** 2)),
            "hidden_mse": float(
                np.mean(
                    (predicted[:, :hidden_output_count] - observed[:, :hidden_output_count]) ** 2
                )
            ),
            "logit_mse": float(
                np.mean(
                    (predicted[:, hidden_output_count:] - observed[:, hidden_output_count:]) ** 2
                )
            ),
            "effect_correlation": effect_correlation(predicted, observed),
            "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
        }
    return {"examples": int(len(observed)), "models": model_metrics}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _study_prompts(count: int) -> list[str]:
    prompts = [
        "Paris is the capital of France, while Rome is the capital of",
        "Water freezes at zero degrees Celsius and boils at",
        "A robot at the left wall should move toward the open",
        "If every dax is a wug and this is a dax, then it is a",
        "The function adds two to its input, so input five gives",
        "First collect the key, then open the locked",
        "The program crashes because the list index is out of",
        "Alice gave the red book to Bob, so the red book belongs to",
    ]
    return [prompts[index % len(prompts)] for index in range(count)]
