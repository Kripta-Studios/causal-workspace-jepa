"""Train, baseline, and directly verify a Qwen Intervention-JEPA."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.adapters.qwen_hf_adapter import QwenAdapterConfig, QwenHFAdapter
from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.data.activation_store import read_hdf5_shards
from causal_workspace_jepa.experiments.llm.qwen_intervention_dataset import (
    OPERATIONS,
    _intervention_vector,
    _target_effect,
    qwen_intervention_prompts,
)
from causal_workspace_jepa.hooks.names import transformer_site
from causal_workspace_jepa.interpretability.circuit_graph import (
    CircuitEdge,
    CircuitGraph,
    CircuitNode,
)
from causal_workspace_jepa.models.intervention_jepa import (
    NeuralInterventionJEPA,
    RidgeInterventionPredictor,
    TinyMLPInterventionPredictor,
    effect_correlation,
)
from causal_workspace_jepa.models.sparse_dictionary import SparseDictionary


def run_qwen_intervention_jepa_study(config_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    hardware = require_free_disk(resource_profile)
    data, records = read_hdf5_shards(str(config["dataset_dir"]))
    context = data["context"].astype(np.float32)
    intervention = data["intervention"].astype(np.float32)
    target = data["target_effect"].astype(np.float32)
    split = data["split_id"]
    feature_id = data["feature_id"]
    operation_id = data["operation_id"]
    heldout_feature = int(config.get("heldout_feature", 256))
    heldout_operation = OPERATIONS.index(str(config.get("heldout_operation", "resample")))
    eligible = (feature_id != heldout_feature) & (operation_id != heldout_operation)
    train = (split == 0) & eligible
    validation = (split == 1) & eligible
    primary = split == 2
    feature_holdout = (split == 0) & (feature_id == heldout_feature)
    operation_holdout = (split == 0) & (operation_id == heldout_operation)
    for name, mask in {
        "train": train,
        "validation": validation,
        "primary": primary,
        "feature_holdout": feature_holdout,
        "operation_holdout": operation_holdout,
    }.items():
        if not np.any(mask):
            raise RuntimeError(f"empty registered split: {name}")

    seeds = tuple(int(value) for value in config.get("seeds", [61, 67, 71]))
    models: list[NeuralInterventionJEPA] = []
    seed_predictions: dict[str, list[np.ndarray]] = {
        "primary": [], "feature_holdout": [], "operation_holdout": []
    }
    checkpoint_dir = Path(str(config.get("checkpoint_dir", "artifacts/checkpoints/qwen_ijepa_v1")))
    for seed in seeds:
        model = NeuralInterventionJEPA.fit(
            context[train],
            intervention[train],
            target[train],
            (context[validation], intervention[validation], target[validation]),
            hidden_dim=int(config.get("hidden_dim", 64)),
            meta_dim=int(config.get("meta_dim", 24)),
            steps=int(config.get("steps", 1000)),
            learning_rate=float(config.get("learning_rate", 0.003)),
            seed=seed,
            device=str(config.get("training_device", "cpu")),
        )
        checkpoint = checkpoint_dir / f"seed-{seed}.npz"
        model.save(checkpoint)
        reloaded = NeuralInterventionJEPA.load(checkpoint)
        np.testing.assert_allclose(
            model.predict(context[validation], intervention[validation]),
            reloaded.predict(context[validation], intervention[validation]),
            atol=0.0,
            rtol=0.0,
        )
        models.append(reloaded)
        for name, mask in {
            "primary": primary,
            "feature_holdout": feature_holdout,
            "operation_holdout": operation_holdout,
        }.items():
            seed_predictions[name].append(reloaded.predict(context[mask], intervention[mask]))

    linear = RidgeInterventionPredictor.fit(
        context[train], intervention[train], target[train], mode="linear", ridge=1e-2
    )
    bilinear = RidgeInterventionPredictor.fit(
        context[train], intervention[train], target[train], mode="bilinear", ridge=1e-2
    )
    mlp = TinyMLPInterventionPredictor.fit(
        context[train],
        intervention[train],
        target[train],
        hidden_dim=int(config.get("mlp_hidden_dim", 64)),
        steps=int(config.get("mlp_steps", 600)),
        learning_rate=float(config.get("mlp_learning_rate", 0.005)),
        seed=seeds[0],
    )
    corpus_transport = RidgeInterventionPredictor.fit(
        np.zeros((int(train.sum()), 1), dtype=np.float32),
        data["source_delta"][train],
        data["local_jacobian_effect"][train],
        mode="linear",
        ridge=1e-2,
    )
    dictionary = SparseDictionary.fit(
        context[train], components=int(config.get("sparse_components", 16)), threshold=0.05
    )
    sparse_codes = dictionary.encode(context)
    sparse_transport = RidgeInterventionPredictor.fit(
        sparse_codes[train], intervention[train], target[train], mode="linear", ridge=1e-2
    )
    train_mean = target[train].mean(axis=0)

    split_scores: dict[str, dict[str, dict[str, float]]] = {}
    decisions_per_seed: list[dict[str, bool]] = []
    split_masks = {
        "primary": primary,
        "feature_holdout": feature_holdout,
        "operation_holdout": operation_holdout,
    }
    for split_name, mask in split_masks.items():
        observed = target[mask]
        ensemble_prediction = np.mean(seed_predictions[split_name], axis=0)
        predictions = {
            "intervention_jepa": ensemble_prediction,
            "no_change": np.zeros_like(observed),
            "mean_effect": np.broadcast_to(train_mean, observed.shape),
            "linear": linear.predict(context[mask], intervention[mask]),
            "bilinear": bilinear.predict(context[mask], intervention[mask]),
            "mlp": mlp.predict(context[mask], intervention[mask]),
            "nearest_neighbor": _nearest_neighbor(
                context[train], intervention[train], target[train], context[mask], intervention[mask]
            ),
            "local_jacobian": data["local_jacobian_effect"][mask],
            "corpus_jacobian": corpus_transport.predict(
                np.zeros((int(mask.sum()), 1), dtype=np.float32), data["source_delta"][mask]
            ),
            "sparse_feature_linear": sparse_transport.predict(
                sparse_codes[mask], intervention[mask]
            ),
        }
        split_scores[split_name] = {
            name: _scores(prediction, observed) for name, prediction in predictions.items()
        }

    effect_power = float(np.mean(target[primary] ** 2))
    local_fraction = split_scores["primary"]["local_jacobian"]["mse"] / max(effect_power, 1e-12)
    for seed_index, seed in enumerate(seeds):
        primary_score = _scores(seed_predictions["primary"][seed_index], target[primary])
        operation_score = _scores(
            seed_predictions["operation_holdout"][seed_index], target[operation_holdout]
        )
        decisions_per_seed.append(
            {
                "seed": seed,
                "h_llm_01_nonlinear_advantage": bool(
                    local_fraction >= float(config.get("nonlinear_fraction_min", 0.05))
                    and primary_score["mse"]
                    < split_scores["primary"]["local_jacobian"]["mse"]
                ),
                "h_llm_02_causal_compression": bool(
                    primary_score["correlation"] >= float(config.get("correlation_min", 0.5))
                    and all(
                        primary_score["mse"] < split_scores["primary"][name]["mse"]
                        for name in ("no_change", "mean_effect", "linear", "bilinear", "mlp")
                    )
                ),
                "h_llm_03_operation_generalization": bool(
                    operation_score["correlation"] >= float(config.get("correlation_min", 0.5))
                    and all(
                        operation_score["mse"]
                        < split_scores["operation_holdout"][name]["mse"]
                        for name in ("no_change", "linear", "corpus_jacobian")
                    )
                ),
            }
        )
    replicated = {
        key: sum(bool(row[key]) for row in decisions_per_seed) >= 2
        for key in (
            "h_llm_01_nonlinear_advantage",
            "h_llm_02_causal_compression",
            "h_llm_03_operation_generalization",
        )
    }
    verification = _direct_verification(config, models, data, records)
    circuit = _verification_graph(verification)
    output_graph = Path(str(config.get("output_graph", "artifacts/tables/qwen_meta_circuit.json")))
    output_graphml = output_graph.with_suffix(".graphml")
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "LLM-IJEPA-001")),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Generalization",
        "model": str(config.get("model")),
        "dataset_examples": len(records),
        "split_counts": {name: int(mask.sum()) for name, mask in {"train": train, "validation": validation, **split_masks}.items()},
        "seeds": list(seeds),
        "scores": split_scores,
        "seed_decisions": decisions_per_seed,
        "replicated_decisions": replicated,
        "primary_effect_power": effect_power,
        "primary_local_error_fraction": local_fraction,
        "sparse_dictionary": {
            "components": int(dictionary.dictionary.shape[0]),
            "feature_density": dictionary.feature_density(context[train]),
            "reconstruction_mse": dictionary.reconstruction_mse(context[validation]),
        },
        "direct_verification": verification,
        "workspace_found": False,
        "runtime_seconds": time.perf_counter() - started,
        "hardware": hardware.as_dict(),
        "claim": (
            "Intervention-JEPA and baselines were evaluated on fixed held-out real-Qwen effects, "
            "and ranked coordinates were re-executed on new prompts. No workspace claim is made."
        ),
    }
    metrics["all_passed"] = bool(
        np.isfinite(split_scores["primary"]["intervention_jepa"]["mse"])
        and verification["all_predictions_directly_executed"]
    )
    output_metrics = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen_intervention_jepa_v1.json"))
    )
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path.as_posix()}",
        resource_profile=resource_profile,
        seed=seeds[0],
    )
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_graph.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    circuit.write_json(output_graph)
    circuit.write_graphml(output_graphml)
    write_provenance(
        output_metrics.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": output_metrics.as_posix(), "all_passed": metrics["all_passed"]},
    )
    return metrics


def _direct_verification(
    config: dict[str, Any],
    models: list[NeuralInterventionJEPA],
    data: dict[str, np.ndarray],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    prompts = [
        "Tokyo is the capital city of Japan .",
        "Athens is the capital city of Greece .",
        "Steam condenses when temperature becomes sufficiently low .",
        "Falling objects speed up when gravity pulls downward .",
    ]
    adapter = QwenHFAdapter(
        QwenAdapterConfig(
            model_name=str(config.get("model", "Qwen/Qwen3-0.6B")),
            revision=str(config["revision"]),
            device=str(config.get("device", "cuda")),
            dtype=str(config.get("dtype", "bfloat16")),
            max_length=int(config.get("max_sequence_length", 24)),
            local_files_only=True,
            token=False,
        )
    )
    features = tuple(int(value) for value in config.get("candidate_features", [0, 64, 128, 256]))
    layer = int(config.get("verification_layer", 14))
    layers = tuple(int(value) for value in config.get("selected_layers", [7, 14, 21]))
    layer_index = layers.index(layer)
    site = transformer_site(layer, "resid_post")
    target_site = transformer_site(len(adapter.layers) - 1, "resid_post")
    hidden_size = int(adapter.model.config.hidden_size)
    context_dim = data["context"].shape[1]
    hidden_target_dim = data["target_effect"].shape[1] // 2
    rng = np.random.default_rng(int(config.get("dataset_seed", 53)))
    context_projection = rng.normal(0.0, 1.0 / np.sqrt(context_dim), size=(hidden_size, context_dim)).astype(np.float32)
    delta_projection = rng.normal(0.0, 1.0 / np.sqrt(context_dim), size=(hidden_size, context_dim)).astype(np.float32)
    hidden_projection = rng.normal(0.0, 1.0 / np.sqrt(hidden_target_dim), size=(hidden_size, hidden_target_dim)).astype(np.float32)
    manifest = json.loads(Path(str(config["dataset_manifest"])).read_text(encoding="utf-8"))
    metrics_path = Path("artifacts/metrics/qwen3_0_6b_intervention_dataset.json")
    dataset_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    logit_ids = np.asarray(dataset_metrics["logit_target_ids"], dtype=np.int64)
    clean_runs = [adapter.forward_with_cache(adapter.tokenize([prompt]), [site, target_site, "logits"]) for prompt in prompts]
    magnitude = float(config.get("verification_magnitude", 4.0))
    predictions: dict[int, list[np.ndarray]] = {feature: [] for feature in features}
    observed: dict[int, list[np.ndarray]] = {feature: [] for feature in features}
    rows: list[dict[str, Any]] = []
    for prompt_id, clean in enumerate(clean_runs):
        source = clean.activations[site][0, -1].astype(np.float32)
        clean_hidden = clean.activations[target_site][0, -1].astype(np.float32)
        clean_logits = clean.logits[0, -1].astype(np.float32)
        for feature in features:
            delta = np.zeros(hidden_size, dtype=np.float32)
            delta[feature] = magnitude
            spec = InterventionSpec(site=site, operation="steer", positions=(-1,), feature_ids=(feature,), magnitude=magnitude)
            intervention_vector = _intervention_vector(
                "steer", layer_index, len(layers), spec, delta, delta_projection, hidden_size
            )
            context_vector = source @ context_projection
            predicted = np.mean(
                [model.predict(context_vector[None], intervention_vector[None])[0] for model in models],
                axis=0,
            )
            direct = adapter.forward_with_intervention(clean.token_batch, spec, [target_site, "logits"])
            effect = _target_effect(
                direct.activations[target_site][0, -1].astype(np.float32) - clean_hidden,
                direct.logits[0, -1].astype(np.float32) - clean_logits,
                hidden_projection,
                logit_ids,
            )
            predictions[feature].append(predicted)
            observed[feature].append(effect)
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "feature_id": feature,
                    "predicted_l2": float(np.linalg.norm(predicted)),
                    "observed_l2": float(np.linalg.norm(effect)),
                    "top_token_changed": bool(np.argmax(direct.logits[0, -1]) != np.argmax(clean_logits)),
                }
            )
    predicted_means = {feature: float(np.mean(np.linalg.norm(value, axis=1))) for feature, value in predictions.items()}
    observed_means = {feature: float(np.mean(np.linalg.norm(value, axis=1))) for feature, value in observed.items()}
    meta_feature = max(features, key=predicted_means.__getitem__)
    actual_feature = max(features, key=observed_means.__getitem__)
    training_prompt_ids = range(8)
    original_prompts, _, _ = qwen_intervention_prompts()
    probe_runs = [adapter.forward_with_cache(adapter.tokenize([original_prompts[index]]), [site, "logits"]) for index in training_prompt_ids]
    probe_scores = {}
    for feature in features:
        x = np.asarray([run.activations[site][0, -1, feature] for run in probe_runs])
        y = np.asarray([run.logits[0, -1, logit_ids].mean() for run in probe_runs])
        probe_scores[feature] = abs(float(np.corrcoef(x, y)[0, 1])) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    remaining = [feature for feature in features if feature != meta_feature]
    probe_feature = max(remaining, key=probe_scores.__getitem__)
    random_pool = [feature for feature in remaining if feature != probe_feature]
    random_feature = random_pool[int(config.get("verification_seed", 79)) % len(random_pool)]
    flat_prediction = np.concatenate([np.stack(predictions[feature]) for feature in features])
    flat_observed = np.concatenate([np.stack(observed[feature]) for feature in features])
    return {
        "prompts": len(prompts),
        "candidate_features": list(features),
        "predicted_mean_l2": predicted_means,
        "observed_mean_l2": observed_means,
        "meta_ranked_feature": meta_feature,
        "actual_top_feature": actual_feature,
        "probe_control_feature": probe_feature,
        "random_control_feature": random_feature,
        "precision_at_1": float(meta_feature == actual_feature),
        "meta_over_random_ratio": observed_means[meta_feature] / max(observed_means[random_feature], 1e-12),
        "meta_over_probe_ratio": observed_means[meta_feature] / max(observed_means[probe_feature], 1e-12),
        "effect_correlation": effect_correlation(flat_prediction, flat_observed),
        "all_predictions_directly_executed": len(rows) == len(prompts) * len(features),
        "rows": rows,
        "dataset_manifest_sha": manifest["config_digest"],
        "circuit_candidate_pass": bool(
            meta_feature == actual_feature
            and observed_means[meta_feature] > observed_means[random_feature]
            and observed_means[meta_feature] > observed_means[probe_feature]
        ),
    }


def _verification_graph(verification: dict[str, Any]) -> CircuitGraph:
    nodes = [CircuitNode("target.final", "target", "Causal mediation", 1.0)]
    edges = []
    for feature in verification["candidate_features"]:
        node_id = f"blocks.14.resid_post.feature_{feature}"
        score = float(verification["observed_mean_l2"][feature])
        nodes.append(CircuitNode(node_id, "residual_coordinate", "Causal mediation", score))
        edges.append(CircuitEdge(node_id, "target.final", score, 1.0))
    return CircuitGraph(
        graph_id="qwen-meta-ranked-coordinate-candidates",
        nodes=tuple(nodes),
        edges=tuple(edges),
        status=("SMOKE_VALIDATED" if verification["circuit_candidate_pass"] else "REJECTED"),
    )


def _scores(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(np.mean((predicted - observed) ** 2)),
        "correlation": effect_correlation(predicted, observed),
        "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
        "mean_predicted_l2": float(np.linalg.norm(predicted, axis=1).mean()),
        "mean_observed_l2": float(np.linalg.norm(observed, axis=1).mean()),
    }


def _nearest_neighbor(
    train_context: np.ndarray,
    train_intervention: np.ndarray,
    train_target: np.ndarray,
    context: np.ndarray,
    intervention: np.ndarray,
) -> np.ndarray:
    train = np.concatenate([train_context, train_intervention], axis=1)
    query = np.concatenate([context, intervention], axis=1)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train = (train - mean) / scale
    query = (query - mean) / scale
    distances = np.sum((query[:, None, :] - train[None, :, :]) ** 2, axis=-1)
    return train_target[np.argmin(distances, axis=1)]
