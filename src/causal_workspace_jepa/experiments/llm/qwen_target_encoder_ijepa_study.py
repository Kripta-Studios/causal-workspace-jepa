"""Genuine target-encoder Intervention-JEPA on behavior-changing Qwen donor patches."""

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
from causal_workspace_jepa.models.intervention_jepa import (
    NeuralInterventionJEPA,
    RidgeInterventionPredictor,
    effect_correlation,
)
from causal_workspace_jepa.models.sparse_dictionary import SparseDictionary
from causal_workspace_jepa.models.target_encoder_intervention_jepa import (
    StandardizedRidgeDecoder,
    TargetEncoderInterventionJEPA,
)


def run_qwen_target_encoder_ijepa_study(config_path: str | Path) -> dict[str, Any]:
    """Train and evaluate a target-encoder JEPA without exposing test outcomes to fitting."""

    import torch

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    hardware = require_free_disk(resource_profile)
    seeds = tuple(int(seed) for seed in config.get("seeds", [311, 313, 317]))
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path.as_posix()}",
        resource_profile=resource_profile,
        seed=seeds[0],
    )
    if provenance.git_dirty:
        raise RuntimeError("LLM-TARGET-IJEPA-001 requires a clean committed worktree")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data, records = read_hdf5_shards(str(config["dataset_dir"]))
    split = data["split_id"].astype(np.int64)
    train = split == 0
    validation = split == 1
    test = split == 2
    for name, mask in {"train": train, "validation": validation, "test": test}.items():
        if not bool(mask.any()):
            raise RuntimeError(f"empty frozen capital split: {name}")
    clean_source = data["clean_source"].astype(np.float32)
    donor_source = data["donor_source"].astype(np.float32)
    source_delta = data["source_delta"].astype(np.float32)
    clean_target = data["clean_target_hidden"].astype(np.float32)
    intervened_target = data["intervened_target_hidden"].astype(np.float32)
    target_effect = data["target_effect"].astype(np.float32)
    answer_count = data["clean_answer_logits"].shape[1]
    context = np.concatenate([clean_source, clean_target], axis=1)
    intervention = np.concatenate([donor_source, source_delta], axis=1)

    checkpoint_dir = Path(
        str(config.get("checkpoint_dir", "artifacts/checkpoints/qwen_target_ijepa_v1"))
    )
    seed_predictions: list[np.ndarray] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        model = TargetEncoderInterventionJEPA.fit(
            clean_source[train],
            donor_source[train],
            clean_target[train],
            intervened_target[train],
            source_delta[train],
            (
                clean_source[validation],
                donor_source[validation],
                clean_target[validation],
                intervened_target[validation],
                source_delta[validation],
            ),
            hidden_dim=int(config.get("hidden_dim", 192)),
            meta_dim=int(config.get("meta_dim", 32)),
            steps=int(config.get("steps", 1200)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
            ema_decay=float(config.get("ema_decay", 0.99)),
            variance_weight=float(config.get("variance_weight", 1.0)),
            covariance_weight=float(config.get("covariance_weight", 0.01)),
            target_consistency_weight=float(config.get("target_consistency_weight", 0.25)),
            std_floor=float(config.get("std_floor", 0.5)),
            validation_interval=int(config.get("validation_interval", 10)),
            patience=int(config.get("patience", 200)),
            seed=seed,
            device=str(config.get("training_device", "cuda")),
        )
        checkpoint = checkpoint_dir / f"seed-{seed}.npz"
        decoder_path = checkpoint_dir / f"seed-{seed}-decoder.npz"
        model.save(checkpoint)
        reloaded = TargetEncoderInterventionJEPA.load(checkpoint)
        target_latent_train = reloaded.target_latent(intervened_target[train])
        clean_latent_train = reloaded.target_latent(clean_target[train])
        decoder = StandardizedRidgeDecoder.fit(
            np.concatenate([target_latent_train, clean_latent_train], axis=1),
            target_effect[train],
            ridge=float(config.get("decoder_ridge", 1.0)),
        )
        decoder.save(decoder_path)
        reloaded_decoder = StandardizedRidgeDecoder.load(decoder_path)
        predicted_latent = reloaded.predict_latent(
            clean_source[test], donor_source[test], clean_target[test], source_delta[test]
        )
        clean_latent_test = reloaded.target_latent(clean_target[test])
        predicted_decoder_input = np.concatenate(
            [predicted_latent, clean_latent_test], axis=1
        )
        predicted_effect = reloaded_decoder.predict(predicted_decoder_input)
        np.testing.assert_array_equal(
            predicted_effect, decoder.predict(predicted_decoder_input)
        )
        oracle_effect = reloaded_decoder.predict(
            np.concatenate(
                [reloaded.target_latent(intervened_target[test]), clean_latent_test], axis=1
            )
        )
        collapse = reloaded.collapse_metrics(
            clean_source[test],
            donor_source[test],
            clean_target[test],
            intervened_target[test],
            source_delta[test],
        )
        predicted_score = _combined_scores(predicted_effect, target_effect[test], data, test)
        oracle_score = _effect_scores(oracle_effect, target_effect[test], answer_count)
        noncollapsed = bool(
            collapse["predicted_mean_dimension_std"]
            >= float(config.get("latent_std_min", 0.10))
            and collapse["target_mean_dimension_std"]
            >= float(config.get("latent_std_min", 0.10))
            and collapse["predicted_effective_rank"]
            >= float(config.get("effective_rank_min", 8.0))
            and collapse["target_effective_rank"]
            >= float(config.get("effective_rank_min", 8.0))
        )
        seed_predictions.append(predicted_effect)
        seed_rows.append(
            {
                "seed": seed,
                "training": reloaded.training_metrics,
                "collapse": collapse,
                "noncollapsed": noncollapsed,
                "test": predicted_score,
                "oracle_target_decoder": oracle_score,
            }
        )
    baseline_predictions = _fit_baselines(
        config,
        context,
        intervention,
        source_delta,
        target_effect,
        train,
        validation,
        test,
        seeds,
    )
    baseline_predictions.update(
        {
            "exact_jvp": data["exact_jvp"][test].astype(np.float32),
            "quadratic_taylor": data["quadratic_taylor"][test].astype(np.float32),
        }
    )
    baseline_scores = {
        name: _combined_scores(prediction, target_effect[test], data, test)
        for name, prediction in baseline_predictions.items()
    }
    ensemble_prediction = np.mean(seed_predictions, axis=0)
    ensemble_score = _combined_scores(ensemble_prediction, target_effect[test], data, test)
    exact_score = baseline_scores["exact_jvp"]
    quadratic_score = baseline_scores["quadratic_taylor"]
    decisions = []
    behavior_margin = float(config.get("behavior_margin", 0.05))
    mse_margin = float(config.get("mse_ratio_max", 0.95))
    for row in seed_rows:
        score = row["test"]
        oracle = row["oracle_target_decoder"]
        decisions.append(
            {
                "seed": row["seed"],
                "h_llm_01b_behavior_nonlinear_advantage": bool(
                    score["answer_candidate_agreement"]
                    >= max(
                        exact_score["answer_candidate_agreement"],
                        quadratic_score["answer_candidate_agreement"],
                    )
                    + behavior_margin
                    and score["logit_normalized_mse"]
                    <= min(
                        exact_score["logit_normalized_mse"],
                        quadratic_score["logit_normalized_mse"],
                    )
                    * mse_margin
                ),
                "h_llm_02_causal_compression": bool(
                    row["noncollapsed"]
                    and oracle["normalized_mse"]
                    <= float(config.get("oracle_normalized_mse_max", 0.75))
                    and score["normalized_mse"]
                    < baseline_scores["mean_effect"]["normalized_mse"]
                    and score["correlation"] >= float(config.get("correlation_min", 0.50))
                ),
                "h_llm_04_entity_generalization": bool(
                    score["answer_candidate_agreement"]
                    >= float(config.get("candidate_agreement_min", 0.50))
                    and score["answer_candidate_agreement"]
                    >= baseline_scores["nearest_neighbor"]["answer_candidate_agreement"]
                    + behavior_margin
                ),
            }
        )
    replicated = {
        key: sum(bool(row[key]) for row in decisions) >= 2
        for key in (
            "h_llm_01b_behavior_nonlinear_advantage",
            "h_llm_02_causal_compression",
            "h_llm_04_entity_generalization",
        )
    }
    outcome_status = (
        "COMPLETED_POSITIVE" if any(replicated.values()) else "COMPLETED_NEGATIVE"
    )
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "LLM-TARGET-IJEPA-001")),
        "status": outcome_status,
        "evidence_level": "Generalization",
        "dataset_id": "LLM-CAPITAL-PATCH-001",
        "dataset_examples": len(records),
        "split_counts": {
            "train": int(train.sum()),
            "validation": int(validation.sum()),
            "test": int(test.sum()),
        },
        "meta_dim": int(config.get("meta_dim", 32)),
        "source_dim": int(clean_source.shape[1]),
        "compression_ratio": float(int(config.get("meta_dim", 32)) / clean_source.shape[1]),
        "seeds": list(seeds),
        "seed_results": seed_rows,
        "seed_decisions": decisions,
        "replicated_decisions": replicated,
        "ensemble_test": ensemble_score,
        "baselines_test": baseline_scores,
        "direct_verification": {
            "all_test_outcomes_directly_executed_on_qwen": True,
            "direct_examples": int(test.sum()),
            "intervention_site": "blocks.21.resid_post",
            "target_site": "blocks.27.resid_post",
            "claim_boundary": (
                "Predicted outcomes are scored against frozen direct Qwen executions. No circuit "
                "candidate or predicted source intervention is introduced by this study."
            ),
        },
        "hardware": hardware.as_dict(),
        "runtime_seconds": float(time.perf_counter() - started),
        "scientific_boundary": (
            "The predictor is a genuine EMA target-encoder JEPA, but its linear outcome decoder "
            "uses predicted and clean target embeddings and is supervised on train entities. A "
            "passing result is bounded entity generalization for one factual prompt family, not "
            "a Qwen circuit, semantic feature, or workspace."
        ),
    }
    output_metrics = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen_target_ijepa_v1.json"))
    )
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output_metrics.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": output_metrics.as_posix(),
            "all_test_outcomes_directly_executed_on_qwen": True,
            "replicated_decisions": replicated,
        },
    )
    return metrics


def _fit_baselines(
    config: dict[str, Any],
    context: np.ndarray,
    intervention: np.ndarray,
    source_delta: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, np.ndarray]:
    train_mean = target[train].mean(axis=0)
    linear = RidgeInterventionPredictor.fit(
        context[train], intervention[train], target[train], mode="linear", ridge=1e-2
    )
    context_projector = _PCAProjector.fit(
        context[train], int(config.get("bilinear_components", 24))
    )
    intervention_projector = _PCAProjector.fit(
        intervention[train], int(config.get("bilinear_components", 24))
    )
    context_code = context_projector.transform(context)
    intervention_code = intervention_projector.transform(intervention)
    bilinear = RidgeInterventionPredictor.fit(
        context_code[train],
        intervention_code[train],
        target[train],
        mode="bilinear",
        ridge=1e-2,
    )
    mlp_prediction = _supervised_mlp_prediction(
        context,
        intervention,
        target,
        train,
        validation,
        test,
        hidden_dim=int(config.get("baseline_hidden_dim", 128)),
        steps=int(config.get("baseline_steps", 800)),
        learning_rate=float(config.get("baseline_learning_rate", 1e-3)),
        patience=int(config.get("baseline_patience", 150)),
        seed=seeds[0],
        device=str(config.get("training_device", "cuda")),
    )
    legacy_predictions = []
    for seed in seeds:
        legacy = NeuralInterventionJEPA.fit(
            context[train],
            intervention[train],
            target[train],
            (context[validation], intervention[validation], target[validation]),
            hidden_dim=int(config.get("legacy_hidden_dim", 128)),
            meta_dim=int(config.get("meta_dim", 32)),
            steps=int(config.get("legacy_steps", 800)),
            learning_rate=float(config.get("legacy_learning_rate", 1e-3)),
            seed=seed,
            device=str(config.get("training_device", "cuda")),
        )
        legacy_predictions.append(legacy.predict(context[test], intervention[test]))
    corpus = RidgeInterventionPredictor.fit(
        np.zeros((int(train.sum()), 1), dtype=np.float32),
        source_delta[train],
        target[train],
        mode="linear",
        ridge=1e-2,
    )
    dictionary = SparseDictionary.fit(
        np.concatenate([context[train], intervention[train]], axis=0),
        components=int(config.get("sparse_components", 32)),
        threshold=float(config.get("sparse_threshold", 0.05)),
    )
    sparse_context = dictionary.encode(context)
    sparse_intervention = dictionary.encode(intervention)
    sparse = RidgeInterventionPredictor.fit(
        sparse_context[train],
        sparse_intervention[train],
        target[train],
        mode="linear",
        ridge=1e-2,
    )
    return {
        "no_change": np.zeros_like(target[test]),
        "mean_effect": np.broadcast_to(train_mean, target[test].shape).copy(),
        "linear": linear.predict(context[test], intervention[test]),
        "pca_bilinear": bilinear.predict(context_code[test], intervention_code[test]),
        "supervised_mlp": mlp_prediction,
        "legacy_conditional_bottleneck": np.mean(legacy_predictions, axis=0),
        "nearest_neighbor": _nearest_neighbor(
            context_code[train],
            intervention_code[train],
            target[train],
            context_code[test],
            intervention_code[test],
        ),
        "corpus_average_transport": corpus.predict(
            np.zeros((int(test.sum()), 1), dtype=np.float32), source_delta[test]
        ),
        "sparse_feature_linear": sparse.predict(
            sparse_context[test], sparse_intervention[test]
        ),
    }


class _PCAProjector:
    def __init__(self, mean: np.ndarray, scale: np.ndarray, components: np.ndarray) -> None:
        self.mean = mean
        self.scale = scale
        self.components = components

    @classmethod
    def fit(cls, values: np.ndarray, components: int) -> "_PCAProjector":
        values = values.astype(np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        normalized = (values - mean) / scale
        _, _, vectors = np.linalg.svd(normalized, full_matrices=False)
        count = min(components, vectors.shape[0])
        return cls(
            mean.astype(np.float32),
            scale.astype(np.float32),
            vectors[:count].astype(np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        normalized = (values.astype(np.float32) - self.mean) / self.scale
        return (normalized @ self.components.T).astype(np.float32)


def _supervised_mlp_prediction(
    context: np.ndarray,
    intervention: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    *,
    hidden_dim: int,
    steps: int,
    learning_rate: float,
    patience: int,
    seed: int,
    device: str,
) -> np.ndarray:
    import torch

    features = np.concatenate([context, intervention], axis=1).astype(np.float32)
    input_mean = features[train].mean(axis=0)
    input_scale = features[train].std(axis=0)
    input_scale[input_scale < 1e-6] = 1.0
    target_mean = target[train].mean(axis=0)
    target_scale = target[train].std(axis=0)
    target_scale[target_scale < 1e-6] = 1.0
    x_train = torch.as_tensor((features[train] - input_mean) / input_scale, device=device)
    y_train = torch.as_tensor((target[train] - target_mean) / target_scale, device=device)
    x_validation = torch.as_tensor(
        (features[validation] - input_mean) / input_scale, device=device
    )
    y_validation = torch.as_tensor(
        (target[validation] - target_mean) / target_scale, device=device
    )
    torch.manual_seed(seed)
    network = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim, target.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    for _step in range(steps):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((network(x_train) - y_train) ** 2)
        loss.backward()
        optimizer.step()
        network.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.mean((network(x_validation) - y_validation) ** 2).cpu()
            )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in network.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("supervised MLP baseline produced no finite checkpoint")
    network.load_state_dict({name: value.to(device) for name, value in best_state.items()})
    network.eval()
    with torch.no_grad():
        normalized = network(
            torch.as_tensor((features[test] - input_mean) / input_scale, device=device)
        ).cpu().numpy()
    return (normalized * target_scale + target_mean).astype(np.float32)


def _effect_scores(
    predicted: np.ndarray, observed: np.ndarray, answer_count: int
) -> dict[str, float]:
    power = max(float(np.mean(observed**2)), 1e-12)
    hidden_power = max(float(np.mean(observed[:, :-answer_count] ** 2)), 1e-12)
    logit_power = max(float(np.mean(observed[:, -answer_count:] ** 2)), 1e-12)
    return {
        "mse": float(np.mean((predicted - observed) ** 2)),
        "normalized_mse": float(np.mean((predicted - observed) ** 2) / power),
        "hidden_normalized_mse": float(
            np.mean((predicted[:, :-answer_count] - observed[:, :-answer_count]) ** 2)
            / hidden_power
        ),
        "logit_normalized_mse": float(
            np.mean((predicted[:, -answer_count:] - observed[:, -answer_count:]) ** 2)
            / logit_power
        ),
        "correlation": effect_correlation(predicted, observed),
    }


def _combined_scores(
    predicted: np.ndarray,
    observed: np.ndarray,
    data: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    answer_count = data["clean_answer_logits"].shape[1]
    result = _effect_scores(predicted, observed, answer_count)
    predicted_candidate = (
        data["clean_answer_logits"][mask] + predicted[:, -answer_count:]
    ).argmax(axis=1)
    direct_candidate = data["intervened_answer_logits"][mask].argmax(axis=1)
    donor_candidate = data["donor_id"][mask]
    result.update(
        {
            "answer_candidate_agreement": float(
                np.mean(predicted_candidate == direct_candidate)
            ),
            "predicted_donor_candidate_rate": float(
                np.mean(predicted_candidate == donor_candidate)
            ),
            "direct_donor_candidate_rate": float(
                np.mean(direct_candidate == donor_candidate)
            ),
        }
    )
    return result


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


__all__ = ["run_qwen_target_encoder_ijepa_study"]
