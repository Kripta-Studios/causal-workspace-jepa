"""Generate a bounded real Qwen intervention-outcome dataset."""

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
from causal_workspace_jepa.common.resources import estimate_activation_bytes, require_free_disk
from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.data.activation_store import write_hdf5_shards
from causal_workspace_jepa.hooks.names import transformer_site


OPERATIONS = ("steer", "zero", "mean", "patch", "resample")


def run_qwen_intervention_dataset(config_path: str | Path) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    hardware = require_free_disk(resource_profile)
    seed = int(config.get("seed", 53))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    prompts, split_ids, families = qwen_intervention_prompts()
    layers = tuple(int(value) for value in config.get("selected_layers", [7, 14, 21]))
    steer_features = tuple(int(value) for value in config.get("steer_features", [0, 64, 128, 256]))
    steer_magnitudes = tuple(float(value) for value in config.get("steer_magnitudes", [-4, 4]))
    replacement_features = tuple(
        int(value) for value in config.get("replacement_features", [0, 64, 128, 256])
    )
    local_epsilon = float(config.get("local_epsilon", 0.05))
    context_dim = int(config.get("context_projection_dim", 32))
    hidden_target_dim = int(config.get("hidden_target_dim", 32))
    logit_target_dim = int(config.get("logit_target_dim", 32))
    expected = len(prompts) * len(layers) * (
        len(steer_features) * len(steer_magnitudes) + 4
    )
    adapter = QwenHFAdapter(
        QwenAdapterConfig(
            model_name=str(config.get("model", "Qwen/Qwen3-0.6B")),
            revision=str(config["revision"]),
            device=str(config.get("device", "cuda")),
            dtype=str(config.get("dtype", "bfloat16")),
            max_length=int(config.get("max_sequence_length", 24)),
            local_files_only=bool(config.get("local_files_only", True)),
            token=False,
        )
    )
    hidden_size = int(adapter.model.config.hidden_size)
    if max((*steer_features, *replacement_features)) >= hidden_size:
        raise ValueError("configured feature id exceeds Qwen hidden size")
    raw_estimate = estimate_activation_bytes(
        examples=expected,
        layers=2,
        positions=1,
        hidden_size=hidden_size,
        bytes_per_value=2,
        overhead_fraction=0.5,
    ) + expected * logit_target_dim * 4
    budget_bytes = int(float(config.get("activation_budget_mb", 128)) * 1024**2)
    if raw_estimate > budget_bytes:
        raise RuntimeError(
            f"BLOCKED_RESOURCE: estimated activation data {raw_estimate} exceeds {budget_bytes}"
        )

    source_sites = [transformer_site(layer, "resid_post") for layer in layers]
    target_site = transformer_site(len(adapter.layers) - 1, "resid_post")
    capture_sites = [*source_sites, target_site, "logits"]
    clean_runs = [adapter.forward_with_cache(adapter.tokenize([prompt]), capture_sites) for prompt in prompts]
    train_indices = np.flatnonzero(np.asarray(split_ids) == 0)
    for site in source_sites:
        mean = np.stack([clean_runs[index].activations[site][0, -1] for index in train_indices]).mean(axis=0)
        adapter.register_mean(site, mean)
        for prompt_index, run in enumerate(clean_runs):
            adapter.register_donor(f"prompt_{prompt_index}", site, run.activations[site])

    train_logits = np.stack([clean_runs[index].logits[0, -1] for index in train_indices])
    top_candidates = np.argsort(train_logits.mean(axis=0))[-logit_target_dim:]
    logit_ids = np.sort(top_candidates).astype(np.int64)
    context_projection = rng.normal(
        0.0, 1.0 / np.sqrt(context_dim), size=(hidden_size, context_dim)
    ).astype(np.float32)
    delta_projection = rng.normal(
        0.0, 1.0 / np.sqrt(context_dim), size=(hidden_size, context_dim)
    ).astype(np.float32)
    hidden_projection = rng.normal(
        0.0, 1.0 / np.sqrt(hidden_target_dim), size=(hidden_size, hidden_target_dim)
    ).astype(np.float32)

    arrays: dict[str, list[np.ndarray | float | int]] = {
        "context": [],
        "intervention": [],
        "source_delta": [],
        "target_effect": [],
        "local_jacobian_effect": [],
        "prompt_id": [],
        "split_id": [],
        "layer": [],
        "operation_id": [],
        "feature_id": [],
        "magnitude": [],
        "clean_top_token": [],
        "intervened_top_token": [],
    }
    records: list[dict[str, Any]] = []
    for prompt_index, clean in enumerate(clean_runs):
        donor_pool = np.flatnonzero(np.asarray(split_ids) == split_ids[prompt_index]).tolist()
        donor_pool = [index for index in donor_pool if index != prompt_index]
        donor_patch = donor_pool[0]
        donor_resample = donor_pool[-1]
        for layer_index, (layer, site) in enumerate(zip(layers, source_sites, strict=True)):
            clean_source = clean.activations[site][0, -1].astype(np.float32)
            clean_hidden = clean.activations[target_site][0, -1].astype(np.float32)
            clean_logits = clean.logits[0, -1].astype(np.float32)
            intervention_rows = _intervention_rows(
                site=site,
                steer_features=steer_features,
                steer_magnitudes=steer_magnitudes,
                replacement_features=replacement_features,
                donor_patch=donor_patch,
                donor_resample=donor_resample,
            )
            for row_index, (operation, spec) in enumerate(intervention_rows):
                edited_source = _edited_source(
                    clean_source,
                    spec,
                    mean=adapter._means[site].detach().float().cpu().numpy(),
                    donor=(
                        adapter._donors[(spec.donor_example_id, site)].detach().float().cpu().numpy()[0, -1]
                        if spec.donor_example_id is not None
                        else None
                    ),
                )
                source_delta = edited_source - clean_source
                direct = adapter.forward_with_intervention(
                    clean.token_batch, spec, [target_site, "logits"]
                )
                effect = _target_effect(
                    direct.activations[target_site][0, -1].astype(np.float32) - clean_hidden,
                    direct.logits[0, -1].astype(np.float32) - clean_logits,
                    hidden_projection,
                    logit_ids,
                )
                epsilon_source = clean_source + local_epsilon * source_delta
                epsilon_id = f"epsilon_{prompt_index}_{layer}_{row_index}"
                adapter.register_donor(epsilon_id, site, epsilon_source)
                epsilon_spec = InterventionSpec(
                    site=site,
                    operation="patch",
                    positions=(-1,),
                    feature_ids=spec.feature_ids,
                    magnitude=1.0,
                    donor_example_id=epsilon_id,
                    seed=seed,
                )
                local = adapter.forward_with_intervention(
                    clean.token_batch, epsilon_spec, [target_site, "logits"]
                )
                local_effect = _target_effect(
                    (local.activations[target_site][0, -1].astype(np.float32) - clean_hidden)
                    / local_epsilon,
                    (local.logits[0, -1].astype(np.float32) - clean_logits) / local_epsilon,
                    hidden_projection,
                    logit_ids,
                )
                intervention_vector = _intervention_vector(
                    operation,
                    layer_index,
                    len(layers),
                    spec,
                    source_delta,
                    delta_projection,
                    hidden_size,
                )
                arrays["context"].append(clean_source @ context_projection)
                arrays["intervention"].append(intervention_vector)
                arrays["source_delta"].append(source_delta @ delta_projection)
                arrays["target_effect"].append(effect)
                arrays["local_jacobian_effect"].append(local_effect)
                arrays["prompt_id"].append(prompt_index)
                arrays["split_id"].append(split_ids[prompt_index])
                arrays["layer"].append(layer)
                arrays["operation_id"].append(OPERATIONS.index(operation))
                arrays["feature_id"].append(
                    spec.feature_ids[0] if operation == "steer" and spec.feature_ids else -1
                )
                arrays["magnitude"].append(spec.magnitude)
                arrays["clean_top_token"].append(int(np.argmax(clean_logits)))
                arrays["intervened_top_token"].append(int(np.argmax(direct.logits[0, -1])))
                records.append(
                    {
                        "example_id": f"qwen-{prompt_index:02d}-{layer:02d}-{row_index:02d}",
                        "prompt_id": prompt_index,
                        "prompt_family": families[prompt_index],
                        "split": ("train", "validation", "test")[split_ids[prompt_index]],
                        "site": site,
                        "target_site": target_site,
                        "intervention": spec.to_dict(),
                        "local_epsilon": local_epsilon,
                    }
                )
    converted = {
        key: np.asarray(value, dtype=_array_dtype(key)) for key, value in arrays.items()
    }
    if converted["context"].shape[0] != expected:
        raise RuntimeError(f"generated {converted['context'].shape[0]} outcomes, expected {expected}")
    config_digest = _config_digest(config)
    local_manifest = write_hdf5_shards(
        str(config.get("output_dir", "data/activations/qwen3_0_6b_interventions_v1")),
        converted,
        records,
        dataset_id=str(config.get("id", "LLM-INTDATA-001")),
        config_digest=config_digest,
        max_shard_mb=float(config.get("max_shard_mb", 256)),
        budget_mb=float(config.get("activation_budget_mb", 128)),
        resume=bool(config.get("resume", True)),
    )
    top_changes = int(np.sum(converted["clean_top_token"] != converted["intervened_top_token"]))
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "LLM-INTDATA-001")),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Causal mediation",
        "model": str(config.get("model")),
        "model_revision": adapter._metadata()["resolved_revision"],
        "seed": seed,
        "outcomes": expected,
        "prompt_counts": {"train": 8, "validation": 2, "test": 2},
        "selected_layers": list(layers),
        "operations": list(OPERATIONS),
        "top_token_changes": top_changes,
        "top_token_change_rate": top_changes / expected,
        "mean_target_effect_l2": float(np.linalg.norm(converted["target_effect"], axis=1).mean()),
        "mean_local_jacobian_mse": float(
            np.mean((converted["local_jacobian_effect"] - converted["target_effect"]) ** 2)
        ),
        "storage": local_manifest,
        "logit_target_ids": logit_ids.tolist(),
        "estimated_bytes_before_capture": raw_estimate,
        "runtime_seconds": time.perf_counter() - started,
        "hardware": hardware.as_dict(),
        "passes": {
            "revision_matches": adapter._metadata()["resolved_revision"] == str(config["revision"]),
            "outcome_count": expected == len(records),
            "all_splits_nonempty": set(converted["split_id"].tolist()) == {0, 1, 2},
            "effects_nonzero": bool(np.all(np.linalg.norm(converted["target_effect"], axis=1) > 0)),
            "within_storage_budget": local_manifest["estimated_uncompressed_bytes"] <= local_manifest["budget_bytes"],
        },
        "claim": (
            "A real, split-controlled Qwen3-0.6B intervention-outcome dataset was generated by "
            "direct execution. This is causal data, not evidence that a meta-model predicts it."
        ),
    }
    metrics["all_passed"] = bool(all(metrics["passes"].values()))
    output_metrics = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen3_0_6b_intervention_dataset.json"))
    )
    output_manifest = Path(
        str(config.get("output_manifest", "data/manifests/qwen3_0_6b_interventions_v1.json"))
    )
    provenance = collect_provenance(
        command=f"python scripts/generate_qwen_interventions.py --config {config_path.as_posix()}",
        resource_profile=resource_profile,
        seed=seed,
    )
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_summary = {
        **local_manifest,
        "model": str(config.get("model")),
        "model_revision": adapter._metadata()["resolved_revision"],
        "prompt_split": {"train": list(range(8)), "validation": [8, 9], "test": [10, 11]},
        "donor_policy": "donors stay within recipient prompt split and exclude the recipient",
        "logit_target_selection": "top mean clean logits on training prompts only",
        "local_data_root": str(config.get("output_dir")),
    }
    for shard in manifest_summary["shards"]:
        shard["path"] = f"{manifest_summary['local_data_root']}/{shard['path']}"
    output_manifest.write_text(
        json.dumps(manifest_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_provenance(
        output_metrics.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": output_metrics.as_posix(), "all_passed": metrics["all_passed"]},
    )
    if not metrics["all_passed"]:
        raise RuntimeError(f"Qwen intervention dataset failed: {metrics['passes']}")
    return metrics


def qwen_intervention_prompts() -> tuple[list[str], list[int], list[str]]:
    prompts = [
        "Paris is the capital city of France .",
        "Madrid is the capital city of Spain .",
        "Rome is the capital city of Italy .",
        "Berlin is the capital city of Germany .",
        "Water becomes solid when temperature falls low .",
        "Metal expands slightly when temperature becomes high .",
        "Plants grow faster when light and water increase .",
        "Objects accelerate when an external force is applied .",
        "Lisbon is the capital city of Portugal .",
        "Vienna is the capital city of Austria .",
        "Ice becomes liquid when temperature rises enough .",
        "Motion slows down when opposing friction becomes stronger .",
    ]
    split_ids = [0] * 8 + [1] * 2 + [2] * 2
    families = ["geography"] * 4 + ["causal_physics"] * 4 + ["geography"] * 2 + ["causal_physics"] * 2
    return prompts, split_ids, families


def _intervention_rows(
    *,
    site: str,
    steer_features: tuple[int, ...],
    steer_magnitudes: tuple[float, ...],
    replacement_features: tuple[int, ...],
    donor_patch: int,
    donor_resample: int,
) -> list[tuple[str, InterventionSpec]]:
    rows = [
        (
            "steer",
            InterventionSpec(
                site=site,
                operation="steer",
                positions=(-1,),
                feature_ids=(feature,),
                magnitude=magnitude,
                seed=53,
            ),
        )
        for feature in steer_features
        for magnitude in steer_magnitudes
    ]
    rows.extend(
        [
            ("zero", InterventionSpec(site=site, operation="zero", positions=(-1,), feature_ids=replacement_features)),
            ("mean", InterventionSpec(site=site, operation="mean", positions=(-1,), feature_ids=replacement_features)),
            (
                "patch",
                InterventionSpec(
                    site=site,
                    operation="patch",
                    positions=(-1,),
                    feature_ids=replacement_features,
                    donor_example_id=f"prompt_{donor_patch}",
                ),
            ),
            (
                "resample",
                InterventionSpec(
                    site=site,
                    operation="resample",
                    positions=(-1,),
                    feature_ids=replacement_features,
                    donor_example_id=f"prompt_{donor_resample}",
                ),
            ),
        ]
    )
    return rows


def _edited_source(
    clean: np.ndarray,
    spec: InterventionSpec,
    *,
    mean: np.ndarray,
    donor: np.ndarray | None,
) -> np.ndarray:
    edited = clean.copy()
    features = list(spec.feature_ids or range(clean.shape[0]))
    if spec.operation == "steer":
        edited[features] += spec.magnitude
    elif spec.operation == "zero":
        edited[features] = 0
    elif spec.operation == "mean":
        edited[features] = mean[features]
    elif spec.operation in {"patch", "resample"}:
        if donor is None:
            raise ValueError("donor operation requires donor activation")
        edited[features] = donor[features]
    else:
        raise ValueError(spec.operation)
    return edited


def _target_effect(
    hidden_delta: np.ndarray,
    logit_delta: np.ndarray,
    hidden_projection: np.ndarray,
    logit_ids: np.ndarray,
) -> np.ndarray:
    return np.concatenate([hidden_delta @ hidden_projection, logit_delta[logit_ids]]).astype(np.float32)


def _intervention_vector(
    operation: str,
    layer_index: int,
    layer_count: int,
    spec: InterventionSpec,
    source_delta: np.ndarray,
    projection: np.ndarray,
    hidden_size: int,
) -> np.ndarray:
    operation_one_hot = np.zeros(len(OPERATIONS), dtype=np.float32)
    operation_one_hot[OPERATIONS.index(operation)] = 1
    layer_one_hot = np.zeros(layer_count, dtype=np.float32)
    layer_one_hot[layer_index] = 1
    feature = float(spec.feature_ids[0] / hidden_size) if operation == "steer" and spec.feature_ids else -1.0
    return np.concatenate(
        [
            operation_one_hot,
            layer_one_hot,
            np.asarray([spec.magnitude, feature, np.linalg.norm(source_delta)], dtype=np.float32),
            source_delta @ projection,
        ]
    ).astype(np.float32)


def _array_dtype(name: str) -> np.dtype[Any]:
    if name in {"context", "intervention", "source_delta", "target_effect", "local_jacobian_effect", "magnitude"}:
        return np.dtype(np.float32)
    return np.dtype(np.int64)


def _config_digest(config: dict[str, Any]) -> str:
    scientific = {key: value for key, value in config.items() if key not in {"status"}}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
