"""Context-paired Qwen causal geometry with pooling and coordinate-gauge controls."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.adapters.qwen_hf_adapter import QwenAdapterConfig, QwenHFAdapter
from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.activation_store import read_hdf5_shards, write_hdf5_shards
from causal_workspace_jepa.experiments.llm.qwen_capital_patch_dataset import (
    CAPITAL_PAIRS,
    SPLIT_COUNTRIES,
)
from causal_workspace_jepa.hooks.names import transformer_site
from causal_workspace_jepa.interpretability.context_causal_geometry import (
    analytic_pooling_counterexample,
    causal_coupling,
    coupling_spectrum,
    euclidean_subspace_overlap,
    pooled_euclidean_overlap,
)
from causal_workspace_jepa.models.intervention_jepa import effect_correlation


def run_qwen_context_geometry_study(config_path: str | Path) -> dict[str, Any]:
    """Compute context-local logit Jacobians and audit pooled causal geometry."""

    import torch

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    hardware = require_free_disk(resource_profile)
    seed = int(config.get("seed", 337))
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path.as_posix()}",
        resource_profile=resource_profile,
        seed=seed,
    )
    if provenance.git_dirty:
        raise RuntimeError("LLM-CONTEXT-GEOMETRY-001 requires a clean committed worktree")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data, patch_records = read_hdf5_shards(str(config["dataset_dir"]))
    adapter = QwenHFAdapter(
        QwenAdapterConfig(
            model_name=str(config.get("model", "Qwen/Qwen3-0.6B")),
            revision=str(config["revision"]),
            device=str(config.get("device", "cuda")),
            dtype="float32",
            max_length=int(config.get("max_sequence_length", 16)),
            local_files_only=bool(config.get("local_files_only", True)),
            preserve_autograd=True,
            attn_implementation=str(config.get("attn_implementation", "eager")),
            token=False,
        )
    )
    adapter.model.eval()
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    source_layer = int(config.get("source_layer", 21))
    source_site = transformer_site(source_layer, "resid_post")
    answer_ids = _single_token_answer_ids(adapter)
    answer_tensor = torch.as_tensor(answer_ids, device=adapter.device)
    prompts = [f"The capital of {country} is" for country, _capital in CAPITAL_PAIRS]
    clean_runs = [
        adapter.forward_with_cache(adapter.tokenize([prompt]), [source_site, "logits"])
        for prompt in prompts
    ]
    expected_bytes = int(len(prompts) * len(answer_ids) * data["clean_source"].shape[1] * 4 * 1.25)
    budget_bytes = int(float(config.get("activation_budget_mb", 16)) * 1024**2)
    if expected_bytes > budget_bytes:
        raise RuntimeError(
            f"BLOCKED_RESOURCE: estimated geometry data {expected_bytes} exceeds {budget_bytes}"
        )
    jacobians = []
    clean_sources = []
    clean_answer_logits = []
    clean_replay_errors = []
    stored_source_errors = []
    for entity_id, run in enumerate(clean_runs):
        source = run.activations[source_site][0, -1].detach().float()
        clean_logits = run.logits[0, -1, answer_tensor].detach().float()
        jacobian, replay_logits = _selected_logit_jacobian(
            adapter,
            run.token_batch,
            source_site=source_site,
            source=source,
            logit_ids=answer_tensor,
        )
        first_row = int(np.flatnonzero(data["recipient_id"] == entity_id)[0])
        stored_source = data["clean_source"][first_row]
        clean_replay_errors.append(
            float((replay_logits - clean_logits).abs().max().detach().cpu())
        )
        stored_source_errors.append(
            float(np.max(np.abs(source.detach().cpu().numpy() - stored_source)))
        )
        jacobians.append(jacobian.detach().float().cpu().numpy())
        clean_sources.append(source.detach().cpu().numpy())
        clean_answer_logits.append(clean_logits.detach().cpu().numpy())
    jacobian_array = np.asarray(jacobians, dtype=np.float32)
    clean_source_array = np.asarray(clean_sources, dtype=np.float32)
    clean_logit_array = np.asarray(clean_answer_logits, dtype=np.float32)
    reconstructed_jvp = np.stack(
        [
            jacobian_array[int(recipient)] @ direction
            for recipient, direction in zip(
                data["recipient_id"], data["source_delta"], strict=True
            )
        ]
    )
    stored_jvp = data["exact_jvp"][:, -len(answer_ids) :]
    jvp_relative = _row_relative_error(reconstructed_jvp, stored_jvp)
    geometry = _analyze_geometry(config, data, jacobian_array, len(answer_ids))
    analytic = analytic_pooling_counterexample()
    numerical_gates = {
        "clean_replacement_replay": max(clean_replay_errors)
        <= float(config.get("clean_replay_max", 1e-5)),
        "stored_source_replay": max(stored_source_errors)
        <= float(config.get("stored_source_max", 1e-6)),
        "jacobian_jvp_median": float(np.median(jvp_relative))
        <= float(config.get("jvp_median_relative_max", 1e-4)),
        "jacobian_jvp_p95": float(np.quantile(jvp_relative, 0.95))
        <= float(config.get("jvp_p95_relative_max", 1e-3)),
        "coupling_diagonal_gauge": geometry["gauge"]["coupling_max_relative_error"]
        <= float(config.get("gauge_relative_max", 1e-5)),
        "analytic_pooling_control": bool(analytic["passed"]),
    }
    hypothesis_decisions = {
        "h_geo_01_real_pooling_illusion": bool(
            geometry["subspace"]["pooled_overlap"]
            >= geometry["subspace"]["matched_mean_overlap"]
            + float(config.get("pooling_gap_min", 0.10))
            and geometry["subspace"]["pooled_permutation_invariance_error"]
            <= float(config.get("pooled_invariance_max", 1e-10))
        ),
        "h_geo_02_context_specific_behavior_transport": bool(
            geometry["transport"]["matched"]["answer_candidate_agreement"]
            >= geometry["transport"]["permutation_null"]["p95_candidate_agreement"]
            + float(config.get("candidate_margin_min", 0.05))
            and geometry["transport"]["matched"]["normalized_mse"]
            <= geometry["transport"]["permutation_null"]["p05_normalized_mse"]
            * float(config.get("permutation_mse_ratio_max", 0.80))
        ),
        "h_geo_03_coordinate_gauge_invariant_coupling": bool(
            geometry["gauge"]["coupling_max_relative_error"]
            <= float(config.get("gauge_relative_max", 1e-5))
            and bool(analytic["passed"])
        ),
    }
    status = (
        "COMPLETED_MIXED"
        if all(numerical_gates.values()) and any(hypothesis_decisions.values())
        else "COMPLETED_NEGATIVE"
        if all(numerical_gates.values())
        else "REJECTED"
    )
    arrays = {
        "logit_jacobian": jacobian_array,
        "clean_source": clean_source_array,
        "clean_answer_logits": clean_logit_array,
        "entity_id": np.arange(len(CAPITAL_PAIRS), dtype=np.int64),
        "split_id": _entity_split_ids(),
    }
    records = [
        {
            "entity_id": entity_id,
            "country": country,
            "capital": capital,
            "prompt": prompts[entity_id],
            "source_site": source_site,
            "output": "36 centered capital-answer logits",
        }
        for entity_id, (country, capital) in enumerate(CAPITAL_PAIRS)
    ]
    storage = write_hdf5_shards(
        str(config.get("output_dir", "data/activations/qwen_context_geometry_v1")),
        arrays,
        records,
        dataset_id=str(config.get("id", "LLM-CONTEXT-GEOMETRY-001")),
        config_digest=_config_digest(config),
        max_shard_mb=float(config.get("max_shard_mb", 16)),
        budget_mb=float(config.get("activation_budget_mb", 16)),
        resume=bool(config.get("resume", True)),
    )
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "LLM-CONTEXT-GEOMETRY-001")),
        "status": status,
        "evidence_level": "Specificity",
        "model": str(config.get("model")),
        "model_revision": adapter._metadata()["resolved_revision"],
        "dataset_id": "LLM-CAPITAL-PATCH-001",
        "patch_examples": len(patch_records),
        "contexts": len(CAPITAL_PAIRS),
        "answer_outputs": len(answer_ids),
        "numerical_gates": numerical_gates,
        "numerical_audit_passed": bool(all(numerical_gates.values())),
        "clean_replacement_replay_max_abs": max(clean_replay_errors),
        "stored_source_replay_max_abs": max(stored_source_errors),
        "jacobian_jvp_median_relative_error": float(np.median(jvp_relative)),
        "jacobian_jvp_p95_relative_error": float(np.quantile(jvp_relative, 0.95)),
        "analytic_pooling_counterexample": analytic,
        "geometry": geometry,
        "hypothesis_decisions": hypothesis_decisions,
        "storage": storage,
        "estimated_bytes_before_capture": expected_bytes,
        "hardware": hardware.as_dict(),
        "runtime_seconds": float(time.perf_counter() - started),
        "identification_assumptions": [
            "Donor residual differences are treated as sampled manifold chords, not arbitrary actions.",
            "Logit Jacobians are local covectors and need not predict finite donor replacements.",
            "The 36-answer contrast endpoint omits non-capital vocabulary logits.",
            "Separate pooled Euclidean spans discard context pairing and are a diagnostic anti-pattern.",
            "The invariant coupling J D^T is coordinate-gauge invariant only when vectors and covectors are transformed dually.",
        ],
        "scientific_boundary": (
            "A passing geometry control diagnoses context pairing or gauge behavior. It does not "
            "identify a semantic feature, compact circuit, J-space, or workspace."
        ),
    }
    output_metrics = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen_context_geometry_v1.json"))
    )
    output_manifest = Path(
        str(config.get("output_manifest", "data/manifests/qwen_context_geometry_v1.json"))
    )
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        **storage,
        "model": str(config.get("model")),
        "model_revision": adapter._metadata()["resolved_revision"],
        "precision": "float32",
        "source_site": source_site,
        "output_definition": "Jacobian of 36 answer logits with respect to final-position source residual",
        "local_data_root": str(config.get("output_dir")),
    }
    for shard in manifest["shards"]:
        shard["path"] = f"{manifest['local_data_root']}/{shard['path']}"
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output_metrics.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": output_metrics.as_posix(),
            "numerical_audit_passed": metrics["numerical_audit_passed"],
            "hypothesis_decisions": hypothesis_decisions,
        },
    )
    if not metrics["numerical_audit_passed"]:
        raise RuntimeError("context geometry failed numerical validity gates")
    return metrics


def _selected_logit_jacobian(
    adapter: QwenHFAdapter,
    batch: Any,
    *,
    source_site: str,
    source: Any,
    logit_ids: Any,
) -> tuple[Any, Any]:
    """Return d(selected logits)/d(source residual) and exact clean replacement output."""

    import torch

    source_layer = int(source_site.split(".")[1])

    def evaluate(replacement: Any) -> Any:
        def source_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            updated = value.clone()
            updated[:, -1, :] = replacement.to(value)
            return (updated, *output[1:]) if isinstance(output, tuple) else updated

        handle = adapter.layers[source_layer].register_forward_hook(source_hook)
        try:
            output = adapter.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
                logits_to_keep=0,
            )
        finally:
            handle.remove()
        return output.logits[0, -1, logit_ids].float()

    source = source.detach().clone().requires_grad_(True)
    replay = evaluate(source)
    jacobian = torch.autograd.functional.jacobian(
        evaluate, source, vectorize=True, strategy="reverse-mode", strict=False
    )
    return jacobian, replay.detach()


def _analyze_geometry(
    config: dict[str, Any],
    data: dict[str, np.ndarray],
    jacobians: np.ndarray,
    answer_count: int,
) -> dict[str, Any]:
    entity_splits = _entity_split_ids()
    test_entities = np.flatnonzero(entity_splits == 2)
    train_entities = np.flatnonzero(entity_splits == 0)
    direction_sets = []
    jacobian_sets = []
    context_spectra: dict[str, Any] = {}
    rank = int(config.get("subspace_rank", 4))
    for entity_id in test_entities:
        rows = (data["split_id"] == 2) & (data["recipient_id"] == entity_id)
        directions = data["source_delta"][rows]
        jacobian = _contrast_jacobian(jacobians[entity_id])
        direction_sets.append(directions)
        jacobian_sets.append(jacobian)
        context_spectra[str(int(entity_id))] = coupling_spectrum(
            causal_coupling(jacobian, directions)
        )
    overlap_matrix = np.asarray(
        [
            [
                euclidean_subspace_overlap(directions, jacobian, rank=rank)
                for jacobian in jacobian_sets
            ]
            for directions in direction_sets
        ]
    )
    matched_overlaps = np.diag(overlap_matrix)
    pooled_overlap = pooled_euclidean_overlap(direction_sets, jacobian_sets, rank=rank)
    rng = np.random.default_rng(int(config.get("permutation_seed", 347)))
    permutation_count = int(config.get("permutations", 256))
    permutations = _derangements(len(test_entities), permutation_count, rng)
    permuted_overlaps = np.asarray(
        [
            np.mean([overlap_matrix[row, column] for row, column in enumerate(permutation)])
            for permutation in permutations
        ]
    )
    first_permuted_pool = pooled_euclidean_overlap(
        direction_sets,
        [jacobian_sets[index] for index in permutations[0]],
        rank=rank,
    )
    test = data["split_id"] == 2
    observed = data["target_effect"][test, -answer_count:]
    matched_prediction = np.stack(
        [
            jacobians[int(recipient)] @ direction
            for recipient, direction in zip(
                data["recipient_id"][test], data["source_delta"][test], strict=True
            )
        ]
    )
    train_mean_jacobian = jacobians[train_entities].mean(axis=0)
    pooled_train_prediction = data["source_delta"][test] @ train_mean_jacobian.T
    permutation_scores = []
    recipient_to_local = {int(entity): index for index, entity in enumerate(test_entities)}
    for permutation in permutations:
        prediction = np.stack(
            [
                jacobians[
                    test_entities[permutation[recipient_to_local[int(recipient)]]]
                ]
                @ direction
                for recipient, direction in zip(
                    data["recipient_id"][test], data["source_delta"][test], strict=True
                )
            ]
        )
        permutation_scores.append(
            _transport_scores(prediction, observed, data, test, answer_count)
        )
    matched_score = _transport_scores(
        matched_prediction, observed, data, test, answer_count
    )
    pooled_train_score = _transport_scores(
        pooled_train_prediction, observed, data, test, answer_count
    )
    diagonal = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=jacobians.shape[-1]))
    coupling_errors = []
    transformed_directions = []
    transformed_jacobians = []
    for directions, jacobian in zip(direction_sets, jacobian_sets, strict=True):
        changed_directions = directions * diagonal
        changed_jacobian = jacobian / diagonal
        reference = causal_coupling(jacobian, directions)
        changed = causal_coupling(changed_jacobian, changed_directions)
        coupling_errors.append(
            float(np.linalg.norm(changed - reference) / max(np.linalg.norm(reference), 1e-12))
        )
        transformed_directions.append(changed_directions)
        transformed_jacobians.append(changed_jacobian)
    transformed_pooled_overlap = pooled_euclidean_overlap(
        transformed_directions, transformed_jacobians, rank=rank
    )
    candidate_values = np.asarray(
        [row["answer_candidate_agreement"] for row in permutation_scores]
    )
    mse_values = np.asarray([row["normalized_mse"] for row in permutation_scores])
    return {
        "subspace": {
            "rank": rank,
            "matched_overlaps": matched_overlaps.tolist(),
            "matched_mean_overlap": float(matched_overlaps.mean()),
            "pooled_overlap": pooled_overlap,
            "permuted_mean_overlap_mean": float(permuted_overlaps.mean()),
            "permuted_mean_overlap_p05": float(np.quantile(permuted_overlaps, 0.05)),
            "permuted_mean_overlap_p95": float(np.quantile(permuted_overlaps, 0.95)),
            "pooled_permutation_invariance_error": abs(
                pooled_overlap - first_permuted_pool
            ),
        },
        "coupling_spectra": context_spectra,
        "transport": {
            "matched": matched_score,
            "train_pooled_jacobian": pooled_train_score,
            "permutation_null": {
                "permutations": permutation_count,
                "mean_candidate_agreement": float(candidate_values.mean()),
                "p05_candidate_agreement": float(np.quantile(candidate_values, 0.05)),
                "p95_candidate_agreement": float(np.quantile(candidate_values, 0.95)),
                "mean_normalized_mse": float(mse_values.mean()),
                "p05_normalized_mse": float(np.quantile(mse_values, 0.05)),
                "p95_normalized_mse": float(np.quantile(mse_values, 0.95)),
            },
        },
        "gauge": {
            "diagonal_condition_number": float(diagonal.max() / diagonal.min()),
            "coupling_max_relative_error": max(coupling_errors),
            "naive_pooled_overlap_original": pooled_overlap,
            "naive_pooled_overlap_transformed": transformed_pooled_overlap,
            "naive_overlap_absolute_change": abs(
                transformed_pooled_overlap - pooled_overlap
            ),
        },
    }


def _transport_scores(
    prediction: np.ndarray,
    observed: np.ndarray,
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    answer_count: int,
) -> dict[str, float]:
    power = max(float(np.mean(observed**2)), 1e-12)
    predicted_candidate = (
        data["clean_answer_logits"][mask] + prediction[:, -answer_count:]
    ).argmax(axis=1)
    direct_candidate = data["intervened_answer_logits"][mask].argmax(axis=1)
    return {
        "normalized_mse": float(np.mean((prediction - observed) ** 2) / power),
        "correlation": effect_correlation(prediction, observed),
        "answer_candidate_agreement": float(
            np.mean(predicted_candidate == direct_candidate)
        ),
    }


def _derangements(count: int, requested: int, rng: np.random.Generator) -> list[np.ndarray]:
    permutations: list[np.ndarray] = []
    identity = np.arange(count)
    while len(permutations) < requested:
        candidate = rng.permutation(count)
        if np.all(candidate != identity):
            permutations.append(candidate)
    return permutations


def _contrast_jacobian(jacobian: np.ndarray) -> np.ndarray:
    return jacobian - jacobian.mean(axis=0, keepdims=True)


def _row_relative_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - target, axis=1) / np.maximum(
        np.linalg.norm(target, axis=1), 1e-8
    )


def _entity_split_ids() -> np.ndarray:
    assignment: dict[str, int] = {}
    for split_id, name in enumerate(("train", "validation", "test")):
        assignment.update({country: split_id for country in SPLIT_COUNTRIES[name]})
    return np.asarray(
        [assignment[country] for country, _capital in CAPITAL_PAIRS], dtype=np.int64
    )


def _single_token_answer_ids(adapter: QwenHFAdapter) -> list[int]:
    answer_ids = [
        int(adapter.tokenizer.encode(f" {capital}", add_special_tokens=False)[0])
        for _country, capital in CAPITAL_PAIRS
    ]
    if any(
        len(adapter.tokenizer.encode(f" {capital}", add_special_tokens=False)) != 1
        for _country, capital in CAPITAL_PAIRS
    ):
        raise RuntimeError("capital geometry requires the frozen single-token answer roster")
    return answer_ids


def _config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


__all__ = ["run_qwen_context_geometry_study"]
