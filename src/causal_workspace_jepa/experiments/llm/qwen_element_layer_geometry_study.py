"""Layerwise causal-control and Jacobian-predictivity studies on factual relations."""

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
from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.data.activation_store import write_hdf5_shards
from causal_workspace_jepa.experiments.llm.qwen_context_geometry_study import (
    _selected_logit_jacobian,
)
from causal_workspace_jepa.hooks.names import transformer_site
from causal_workspace_jepa.models.intervention_jepa import effect_correlation


ELEMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("Hydrogen", "H"),
    ("Helium", "He"),
    ("Lithium", "Li"),
    ("Beryllium", "Be"),
    ("Boron", "B"),
    ("Carbon", "C"),
    ("Nitrogen", "N"),
    ("Oxygen", "O"),
    ("Fluorine", "F"),
    ("Neon", "Ne"),
    ("Sodium", "Na"),
    ("Magnesium", "Mg"),
    ("Aluminum", "Al"),
    ("Silicon", "Si"),
    ("Phosphorus", "P"),
    ("Sulfur", "S"),
    ("Chlorine", "Cl"),
    ("Argon", "Ar"),
    ("Potassium", "K"),
    ("Calcium", "Ca"),
    ("Scandium", "Sc"),
    ("Titanium", "Ti"),
    ("Vanadium", "V"),
    ("Chromium", "Cr"),
    ("Manganese", "Mn"),
    ("Iron", "Fe"),
    ("Cobalt", "Co"),
    ("Nickel", "Ni"),
    ("Copper", "Cu"),
    ("Strontium", "Sr"),
    ("Gallium", "Ga"),
    ("Germanium", "Ge"),
    ("Arsenic", "As"),
    ("Selenium", "Se"),
    ("Bromine", "Br"),
    ("Krypton", "Kr"),
)

ELEMENT_SPLIT_IDS = np.asarray(
    [
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        2,
        0,
        2,
        2,
        2,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
    ],
    dtype=np.int64,
)

STATE_PAIRS: tuple[tuple[str, str], ...] = (
    ("Alaska", "AK"),
    ("Arizona", "AZ"),
    ("Arkansas", "AR"),
    ("California", "CA"),
    ("Connecticut", "CT"),
    ("Delaware", "DE"),
    ("Florida", "FL"),
    ("Georgia", "GA"),
    ("Hawaii", "HI"),
    ("Idaho", "ID"),
    ("Illinois", "IL"),
    ("Kansas", "KS"),
    ("Kentucky", "KY"),
    ("Louisiana", "LA"),
    ("Maine", "ME"),
    ("Maryland", "MD"),
    ("Michigan", "MI"),
    ("Montana", "MT"),
    ("Nebraska", "NE"),
    ("Nevada", "NV"),
    ("New Hampshire", "NH"),
    ("New Jersey", "NJ"),
    ("New Mexico", "NM"),
    ("New York", "NY"),
    ("North Carolina", "NC"),
    ("Oklahoma", "OK"),
    ("Oregon", "OR"),
    ("Pennsylvania", "PA"),
    ("South Carolina", "SC"),
    ("South Dakota", "SD"),
    ("Tennessee", "TN"),
    ("Utah", "UT"),
    ("Vermont", "VT"),
    ("Washington", "WA"),
    ("West Virginia", "WV"),
    ("Wisconsin", "WI"),
)

STATE_SPLIT_IDS = np.asarray(
    [
        0,
        1,
        0,
        0,
        0,
        2,
        1,
        0,
        1,
        0,
        2,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        2,
        0,
        0,
        2,
        0,
        0,
        1,
        1,
    ],
    dtype=np.int64,
)


def _relation_spec(config: dict[str, Any]) -> dict[str, Any]:
    relation = str(config.get("relation", "element_symbol"))
    if relation == "element_symbol":
        return {
            "name": relation,
            "pairs": ELEMENT_PAIRS,
            "split_ids": ELEMENT_SPLIT_IDS,
            "prompt_template": "The chemical symbol for {entity} is",
            "donor_prefix": "element",
            "calibration_boundary": {
                "excluded_elements": ["Gold", "Silver", "Tin", "Lead"],
                "selected_transition": [21, 24],
                "registered_entities_used_for_calibration": False,
            },
            "scientific_boundary": (
                "Late factual crystallization and attribution-patching nonlinearity are prior "
                "art. This study tests the narrower coupling between directly executed donor "
                "control and a layerwise inversion in local-versus-population finite-effect "
                "predictivity. It does not identify a feature, circuit, J-space, or workspace."
            ),
        }
    if relation == "us_state_postal_abbreviation":
        return {
            "name": relation,
            "pairs": STATE_PAIRS,
            "split_ids": STATE_SPLIT_IDS,
            "prompt_template": "The postal abbreviation for {entity} is",
            "donor_prefix": "state",
            "calibration_boundary": {
                "selection_seed": 521,
                "split_seed": 523,
                "model_forwards_before_preregistration": 0,
                "layer_grid_inherited_from": "LLM-ELEMENT-LAYER-GEOMETRY-001",
            },
            "scientific_boundary": (
                "Late factual crystallization, population Jacobians, and HVP corrections are "
                "prior art. This untouched relation prospectively tests whether the relative "
                "utility of population transport changes sign at a direct causal-control boundary. "
                "It does not identify a feature, circuit, J-space, or workspace."
            ),
        }
    if relation == "us_state_postal_abbreviation_one_shot":
        return {
            "name": relation,
            "pairs": STATE_PAIRS,
            "split_ids": STATE_SPLIT_IDS,
            "prompt_template": (
                "Example: The postal abbreviation for District of Columbia is DC. "
                "The postal abbreviation for {entity} is"
            ),
            "donor_prefix": "state_one_shot",
            "calibration_boundary": {
                "prompt_candidates": 5,
                "prompt_calibration_entities": 13,
                "selected_prompt_calibration_accuracy": 1.0,
                "target_prompt_forwards_before_preregistration": 0,
                "layer_grid_inherited_from": "LLM-ELEMENT-LAYER-GEOMETRY-001",
            },
            "scientific_boundary": (
                "Population Jacobians, late crystallization, and HVP corrections are prior art. "
                "This new prompt prospectively tests whether the empirical onset of direct donor "
                "control aligns with the onset of population-over-local/HVP advantage. It does "
                "not identify a feature, circuit, J-space, or workspace."
            ),
        }
    raise ValueError(f"unsupported factual relation: {relation}")


def run_qwen_element_layer_geometry_study(config_path: str | Path) -> dict[str, Any]:
    """Execute a registered factual-relation patch/Jacobian study."""

    import torch

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    relation = _relation_spec(config)
    relation_pairs: tuple[tuple[str, str], ...] = relation["pairs"]
    split_ids: np.ndarray = relation["split_ids"]
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    hardware = require_free_disk(resource_profile)
    seed = int(config.get("seed", 463))
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path.as_posix()}",
        resource_profile=resource_profile,
        seed=seed,
    )
    if provenance.git_dirty:
        raise RuntimeError(f"{config['id']} requires a clean committed worktree")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = QwenHFAdapter(
        QwenAdapterConfig(
            model_name=str(config.get("model", "Qwen/Qwen3-0.6B")),
            revision=str(config["revision"]),
            device=str(config.get("device", "cuda")),
            dtype="float32",
            max_length=int(config.get("max_sequence_length", 24)),
            local_files_only=bool(config.get("local_files_only", True)),
            preserve_autograd=True,
            attn_implementation=str(config.get("attn_implementation", "eager")),
            token=False,
        )
    )
    adapter.model.eval()
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    layers = tuple(int(value) for value in config.get("layers", [18, 21, 24, 26]))
    sites = tuple(transformer_site(layer, "resid_post") for layer in layers)
    prompts = [str(relation["prompt_template"]).format(entity=entity) for entity, _answer in relation_pairs]
    answer_ids = _answer_ids(adapter, relation_pairs, relation_name=str(relation["name"]))
    answer_tensor = torch.as_tensor(answer_ids, device=adapter.device)
    clean_runs = [
        adapter.forward_with_cache(adapter.tokenize([prompt]), [*sites, "logits"])
        for prompt in prompts
    ]
    clean_sources = np.stack(
        [
            np.stack([_numpy(run.activations[site][0, -1]) for run in clean_runs])
            for site in sites
        ]
    )
    clean_answer_logits = np.stack(
        [_numpy(run.logits[0, -1, answer_tensor]) for run in clean_runs]
    )
    clean_top_tokens = np.asarray(
        [int(run.logits[0, -1].argmax().detach().cpu()) for run in clean_runs], dtype=np.int64
    )
    jacobians, replay_errors = _full_jacobians(
        adapter, clean_runs, sites, answer_tensor
    )
    pairs = _within_split_pairs(split_ids)
    recipient_ids = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    donor_ids = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    patch_split_ids = split_ids[recipient_ids]
    layer_count = len(layers)
    outcome_count = len(pairs)
    answer_count = len(answer_ids)
    hidden_size = clean_sources.shape[-1]
    expected_bytes = int(
        4
        * (
            layer_count * len(relation_pairs) * answer_count * hidden_size
            + layer_count * outcome_count * hidden_size
            + 5 * layer_count * outcome_count * answer_count
        )
        * 1.25
    )
    budget_bytes = int(float(config.get("activation_budget_mb", 96)) * 1024**2)
    if expected_bytes > budget_bytes:
        raise RuntimeError(
            f"BLOCKED_RESOURCE: estimated {relation['name']} geometry "
            f"{expected_bytes} exceeds {budget_bytes}"
        )
    source_delta = np.empty((layer_count, outcome_count, hidden_size), dtype=np.float32)
    intervened_logits = np.empty(
        (layer_count, outcome_count, answer_count), dtype=np.float32
    )
    intervened_top = np.empty((layer_count, outcome_count), dtype=np.int64)
    local_prediction = np.empty_like(intervened_logits)
    quadratic_prediction = np.empty_like(intervened_logits)
    central_prediction = np.empty_like(intervened_logits)
    source_semantic_error = np.empty((layer_count, outcome_count), dtype=np.float32)
    central_relative_error = np.empty((layer_count, outcome_count), dtype=np.float32)
    epsilon = float(config.get("central_epsilon", 0.125))
    for layer_index, (layer, site) in enumerate(zip(layers, sites, strict=True)):
        for entity_id, run in enumerate(clean_runs):
            adapter.register_donor(
                f"{relation['donor_prefix']}_{layer}_{entity_id}", site, run.activations[site]
            )
        for row, (recipient_id, donor_id) in enumerate(pairs):
            clean = clean_runs[recipient_id]
            recipient_source = clean.activations[site][0, -1].detach().float()
            donor_source = clean_runs[donor_id].activations[site][0, -1].detach().float()
            direction = donor_source - recipient_source
            direct = adapter.forward_with_intervention(
                clean.token_batch,
                InterventionSpec(
                    site=site,
                    operation="patch",
                    positions=(-1,),
                    donor_example_id=f"{relation['donor_prefix']}_{layer}_{donor_id}",
                    seed=seed,
                ),
                [site, "logits"],
            )
            direct_logits = direct.logits[0, -1, answer_tensor].detach().float()
            local = torch.as_tensor(
                jacobians[layer_index, recipient_id], device=adapter.device
            ) @ direction
            evaluate = _chord_function(
                adapter,
                clean.token_batch,
                source_site=site,
                source=recipient_source,
                direction=direction,
                logit_ids=answer_tensor,
            )
            with torch.no_grad():
                plus = evaluate(torch.tensor(epsilon, device=adapter.device))
                minus = evaluate(torch.tensor(-epsilon, device=adapter.device))
            clean_logits = clean.logits[0, -1, answer_tensor].detach().float()
            central = (plus - minus) / (2.0 * epsilon)
            second = (plus - 2.0 * clean_logits + minus) / (epsilon**2)
            source_delta[layer_index, row] = _numpy(direction)
            intervened_logits[layer_index, row] = _numpy(direct_logits)
            intervened_top[layer_index, row] = int(
                direct.logits[0, -1].argmax().detach().cpu()
            )
            local_prediction[layer_index, row] = _numpy(local)
            quadratic_prediction[layer_index, row] = _numpy(local + 0.5 * second)
            central_prediction[layer_index, row] = _numpy(central)
            direct_source = direct.activations[site][0, -1].detach().float()
            source_semantic_error[layer_index, row] = float(
                (direct_source - donor_source).abs().max().cpu()
            )
            central_relative_error[layer_index, row] = float(
                (central - local).norm().cpu() / local.norm().clamp_min(1e-8).cpu()
            )
    arrays = {
        "clean_source": clean_sources,
        "logit_jacobian": jacobians,
        "clean_answer_logits": clean_answer_logits,
        "clean_top_token": clean_top_tokens,
        "source_delta": source_delta,
        "intervened_answer_logits": intervened_logits,
        "intervened_top_token": intervened_top,
        "local_prediction": local_prediction,
        "quadratic_prediction": quadratic_prediction,
        "central_prediction": central_prediction,
        "source_semantic_error": source_semantic_error,
        "central_relative_error": central_relative_error,
        "recipient_id": recipient_ids,
        "donor_id": donor_ids,
        "split_id": patch_split_ids,
        "entity_split_id": split_ids,
        "answer_token_id": np.asarray(answer_ids, dtype=np.int64),
        "layer": np.asarray(layers, dtype=np.int64),
    }
    analysis = _analyze(config, arrays, answer_ids, layers, replay_errors)
    stored_arrays = {name: value[None] for name, value in arrays.items()}
    records = [
        {
            "record_type": f"complete_{relation['name']}_layer_grid",
            "outcomes_per_layer": outcome_count,
            "layers": list(layers),
            "operation": "full_residual_donor_patch_at_final_prompt_position",
            "index_arrays": ["recipient_id", "donor_id", "split_id"],
        }
    ]
    storage = write_hdf5_shards(
        str(config.get("output_dir", "data/activations/qwen_relation_layer_geometry_v1")),
        stored_arrays,
        records,
        dataset_id=str(config.get("id", "LLM-RELATION-LAYER-GEOMETRY-001")),
        config_digest=_config_digest(config),
        max_shard_mb=float(config.get("max_shard_mb", 64)),
        budget_mb=float(config.get("activation_budget_mb", 96)),
        resume=bool(config.get("resume", True)),
    )
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "LLM-RELATION-LAYER-GEOMETRY-001")),
        "status": analysis["status"],
        "evidence_level": (
            "Generalization" if analysis["status"].startswith("COMPLETED") else "Availability"
        ),
        "model": str(config.get("model")),
        "model_revision": adapter._metadata()["resolved_revision"],
        "precision": "float32",
        "prompt_template": relation["prompt_template"],
        "relation": relation["name"],
        "entities": len(relation_pairs),
        "outcomes_per_layer": outcome_count,
        "layers": list(layers),
        "split_entity_ids": {
            name: np.flatnonzero(split_ids == index).tolist()
            for index, name in enumerate(("train", "validation", "test"))
        },
        "calibration_boundary": relation["calibration_boundary"],
        "analysis": analysis,
        "storage": storage,
        "estimated_bytes_before_capture": expected_bytes,
        "hardware": hardware.as_dict(),
        "runtime_seconds": float(time.perf_counter() - started),
        "scientific_boundary": relation["scientific_boundary"],
    }
    output_metrics = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen_relation_layer_geometry_v1.json"))
    )
    output_manifest = Path(
        str(config.get("output_manifest", "data/manifests/qwen_relation_layer_geometry_v1.json"))
    )
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", "utf-8")
    manifest = {
        **storage,
        "model": metrics["model"],
        "model_revision": metrics["model_revision"],
        "precision": "float32",
        "layers": list(layers),
        "local_data_root": str(config.get("output_dir")),
        "entity_split_ids": split_ids.tolist(),
    }
    for shard in manifest["shards"]:
        shard["path"] = f"{manifest['local_data_root']}/{shard['path']}"
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    write_provenance(
        output_metrics.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": output_metrics.as_posix(),
            "numerical_audit_passed": analysis["numerical_audit_passed"],
            "hypothesis_decisions": analysis["hypothesis_decisions"],
        },
    )
    if not analysis["numerical_audit_passed"]:
        raise RuntimeError(f"{relation['name']} layer geometry failed numerical validity gates")
    return metrics


def _answer_ids(
    adapter: QwenHFAdapter,
    relation_pairs: tuple[tuple[str, str], ...] = ELEMENT_PAIRS,
    *,
    relation_name: str = "element_symbol",
) -> list[int]:
    result = []
    for _entity, answer in relation_pairs:
        token_ids = adapter.tokenizer.encode(f" {answer}", add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(
                f"registered {relation_name} answer is not one token: {answer} -> {token_ids}"
            )
        result.append(int(token_ids[0]))
    if len(set(result)) != len(result):
        raise RuntimeError(f"{relation_name} answer token IDs must be unique")
    return result


def _within_split_pairs(split_ids: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(
        (recipient, donor)
        for split in range(3)
        for recipient in np.flatnonzero(split_ids == split)
        for donor in np.flatnonzero(split_ids == split)
        if recipient != donor
    )


def _full_jacobians(
    adapter: QwenHFAdapter,
    clean_runs: list[Any],
    sites: tuple[str, ...],
    answer_tensor: Any,
) -> tuple[np.ndarray, list[list[float]]]:
    layers = []
    replay_errors: list[list[float]] = []
    for site in sites:
        rows = []
        errors = []
        for run in clean_runs:
            source = run.activations[site][0, -1].detach().float()
            jacobian, replay = _selected_logit_jacobian(
                adapter,
                run.token_batch,
                source_site=site,
                source=source,
                logit_ids=answer_tensor,
            )
            clean = run.logits[0, -1, answer_tensor].detach().float()
            rows.append(_numpy(jacobian))
            errors.append(float((replay - clean).abs().max().detach().cpu()))
        layers.append(np.stack(rows))
        replay_errors.append(errors)
    return np.stack(layers), replay_errors


def _chord_function(
    adapter: QwenHFAdapter,
    batch: Any,
    *,
    source_site: str,
    source: Any,
    direction: Any,
    logit_ids: Any,
):
    source_layer = int(source_site.split(".")[1])

    def evaluate(scale: Any) -> Any:
        replacement = source + scale.to(source) * direction

        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            updated = value.clone()
            updated[:, -1, :] = replacement.to(value)
            return (updated, *output[1:]) if isinstance(output, tuple) else updated

        handle = adapter.layers[source_layer].register_forward_hook(hook)
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

    return evaluate


def _analyze(
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    answer_ids: list[int],
    layers: tuple[int, ...],
    replay_errors: list[list[float]],
) -> dict[str, Any]:
    numerical_by_layer: dict[str, Any] = {}
    layer_scores: dict[str, Any] = {}
    behavior: dict[str, Any] = {}
    train_entities = np.flatnonzero(arrays["entity_split_id"] == 0)
    clean_correct = arrays["clean_top_token"] == arrays["answer_token_id"]
    clean_accuracy = {
        name: float(clean_correct[arrays["entity_split_id"] == split].mean())
        for split, name in enumerate(("train", "validation", "test"))
    }
    for layer_index, layer in enumerate(layers):
        relative = arrays["central_relative_error"][layer_index]
        numerical_by_layer[str(layer)] = {
            "clean_replacement_replay_max_abs": max(replay_errors[layer_index]),
            "direct_source_semantic_max_abs": float(
                arrays["source_semantic_error"][layer_index].max()
            ),
            "jvp_central_median_relative_error": float(np.median(relative)),
            "jvp_central_p95_relative_error": float(np.quantile(relative, 0.95)),
            "passed": bool(
                max(replay_errors[layer_index]) <= float(config.get("clean_replay_max", 1e-5))
                and arrays["source_semantic_error"][layer_index].max()
                <= float(config.get("source_semantic_max", 1e-6))
                and np.median(relative)
                <= float(config.get("central_median_relative_max", 0.05))
                and np.quantile(relative, 0.95)
                <= float(config.get("central_p95_relative_max", 0.15))
            ),
        }
        population_jacobian = arrays["logit_jacobian"][layer_index, train_entities].mean(axis=0)
        population_prediction = arrays["source_delta"][layer_index] @ population_jacobian.T
        train_rows = arrays["split_id"] == 0
        direct_effect = (
            arrays["intervened_answer_logits"][layer_index]
            - arrays["clean_answer_logits"][arrays["recipient_id"]]
        )
        mean_effect = direct_effect[train_rows].mean(axis=0, keepdims=True)
        scores_by_split: dict[str, Any] = {}
        for split, split_name in enumerate(("train", "validation", "test")):
            mask = arrays["split_id"] == split
            scores_by_split[split_name] = {
                "no_change": _scores(np.zeros_like(direct_effect[mask]), direct_effect[mask], arrays, layer_index, mask),
                "mean_effect": _scores(np.repeat(mean_effect, mask.sum(), axis=0), direct_effect[mask], arrays, layer_index, mask),
                "local_jacobian": _scores(arrays["local_prediction"][layer_index, mask], direct_effect[mask], arrays, layer_index, mask),
                "quadratic_taylor": _scores(arrays["quadratic_prediction"][layer_index, mask], direct_effect[mask], arrays, layer_index, mask),
                "train_population_jacobian": _scores(population_prediction[mask], direct_effect[mask], arrays, layer_index, mask),
            }
        row_null = _row_permutation_null(
            config,
            population_jacobian,
            arrays,
            layer_index,
            direct_effect,
            split=2,
            seed=int(config.get("row_permutation_seed", 479)) + layer,
        )
        curve = _averaging_curve(
            config,
            arrays["logit_jacobian"][layer_index],
            train_entities,
            arrays,
            layer_index,
            direct_effect,
            split=2,
            seed=int(config.get("averaging_seed", 487)) + layer,
        )
        layer_scores[str(layer)] = {
            "splits": scores_by_split,
            "row_permutation_null_test": row_null,
            "averaging_curve_test": curve,
            "population_jacobian_frobenius_norm": float(np.linalg.norm(population_jacobian)),
            "context_jacobian_relative_dispersion": float(
                np.mean((arrays["logit_jacobian"][layer_index] - arrays["logit_jacobian"][layer_index].mean(0)) ** 2)
                / max(float(np.mean(arrays["logit_jacobian"][layer_index] ** 2)), 1e-12)
            ),
        }
        donor_tokens = arrays["answer_token_id"][arrays["donor_id"]]
        candidate_direct = arrays["intervened_answer_logits"][layer_index].argmax(axis=1)
        behavior[str(layer)] = {
            split_name: {
                "full_vocab_donor_token_transfer": float(
                    np.mean(arrays["intervened_top_token"][layer_index, arrays["split_id"] == split] == donor_tokens[arrays["split_id"] == split])
                ),
                "answer_candidate_donor_transfer": float(
                    np.mean(candidate_direct[arrays["split_id"] == split] == arrays["donor_id"][arrays["split_id"] == split])
                ),
                "full_vocab_top_token_change": float(
                    np.mean(
                        arrays["intervened_top_token"][layer_index, arrays["split_id"] == split]
                        != arrays["clean_top_token"][arrays["recipient_id"][arrays["split_id"] == split]]
                    )
                ),
            }
            for split, split_name in enumerate(("train", "validation", "test"))
        }
    numerical_passed = bool(all(value["passed"] for value in numerical_by_layer.values()))
    decision_family = str(config.get("decision_family", "element_inversion"))
    behavior_floor = float(config.get("clean_behavior_accuracy_min", 0.0))
    behavior_eligible = bool(
        clean_accuracy["validation"] >= behavior_floor
        and clean_accuracy["test"] >= behavior_floor
    )
    transition_pass = bool(behavior_eligible and _transition_decision(config, behavior))
    semantic_pass = bool(behavior_eligible and _semantic_decision(config, layer_scores))
    frozen_confirmation: dict[str, Any]
    population_advantage = _population_advantage_details(layer_scores, behavior)
    boundary_alignment = _boundary_alignment_details(config, population_advantage)
    if decision_family == "boundary_conditioned_population":
        alignment_pass = bool(
            behavior_eligible
            and _boundary_alignment_decision(config, population_advantage, boundary_alignment)
        )
        transition_pass = bool(
            behavior_eligible and _terminal_transition_decision(config, behavior)
        )
        semantic_pass = bool(
            behavior_eligible
            and _boundary_semantic_decision(config, layer_scores, boundary_alignment)
        )
        element = json.loads(
            Path(str(config["element_confirmation_metrics"])).read_text("utf-8")
        )
        element_analysis = element["analysis"]
        element_details = _population_advantage_details(
            element_analysis["layer_scores"], element_analysis["behavior_by_layer"]
        )
        element_boundaries = _boundary_alignment_details(config, element_details)
        element_decisions = element_analysis["hypothesis_decisions"]
        element_pass = bool(
            element_decisions["h_llm_08_late_causal_control_transition"]
            and element_decisions["h_geo_09_late_population_semantic_specificity"]
            and _boundary_sign_equality_decision(
                config, element_details, element_boundaries
            )
        )
        decisions = {
            "h_llm_12_one_shot_late_causal_control": transition_pass,
            "h_geo_12_control_population_boundary_alignment": alignment_pass,
            "h_geo_13_boundary_population_semantic_specificity": semantic_pass,
            "h_cross_05_boundary_alignment_cross_relation": bool(
                element_pass and transition_pass and alignment_pass and semantic_pass
            ),
        }
        frozen_confirmation = {
            "element_metrics": str(config["element_confirmation_metrics"]),
            "element_boundary_alignment_passed": element_pass,
            "element_boundary_alignment": element_boundaries,
        }
    elif decision_family == "control_conditioned_population":
        association_pass = bool(
            behavior_eligible
            and _control_population_association_decision(config, population_advantage)
        )
        element = json.loads(
            Path(str(config["element_confirmation_metrics"])).read_text("utf-8")
        )
        element_decisions = element["analysis"]["hypothesis_decisions"]
        element_pass = bool(
            element_decisions["h_llm_08_late_causal_control_transition"]
            and element_decisions["h_geo_09_late_population_semantic_specificity"]
        )
        decisions = {
            "h_llm_10_state_late_causal_control_transition": transition_pass,
            "h_geo_10_control_conditioned_population_advantage": association_pass,
            "h_geo_11_state_late_population_semantic_specificity": semantic_pass,
            "h_cross_04_control_conditioned_transport_cross_relation": bool(
                element_pass and transition_pass and association_pass and semantic_pass
            ),
        }
        frozen_confirmation = {
            "element_metrics": str(config["element_confirmation_metrics"]),
            "element_h_llm_08_and_h_geo_09_passed": element_pass,
        }
    else:
        inversion_pass = _inversion_decision(config, layer_scores)
        capital = json.loads(
            Path(str(config["capital_confirmation_metrics"])).read_text("utf-8")
        )
        capital_pass = bool(all(capital["hypothesis_decisions"].values()))
        decisions = {
            "h_llm_08_late_causal_control_transition": transition_pass,
            "h_geo_08_local_population_predictivity_inversion": inversion_pass,
            "h_geo_09_late_population_semantic_specificity": semantic_pass,
            "h_cross_03_population_transport_cross_relation": bool(
                capital_pass and transition_pass and inversion_pass and semantic_pass
            ),
        }
        frozen_confirmation = {
            "capital_metrics": str(config["capital_confirmation_metrics"]),
            "capital_confirmation_passed": capital_pass,
        }
    if not numerical_passed:
        status = "REJECTED_NUMERICAL_GATE"
    elif not behavior_eligible:
        status = "REJECTED_BEHAVIOR_GATE"
    elif all(decisions.values()):
        status = "COMPLETED_POSITIVE"
    elif any(decisions.values()):
        status = "COMPLETED_MIXED"
    else:
        status = "COMPLETED_NEGATIVE"
    return {
        "status": status,
        "numerical_audit_passed": numerical_passed,
        "numerical_by_layer": numerical_by_layer,
        "clean_full_vocab_accuracy_by_split": clean_accuracy,
        "clean_behavior_accuracy_min": behavior_floor,
        "behavior_eligible": behavior_eligible,
        "behavior_by_layer": behavior,
        "layer_scores": layer_scores,
        "population_advantage_by_layer": population_advantage,
        "boundary_alignment_by_split": boundary_alignment,
        "frozen_confirmation": frozen_confirmation,
        "hypothesis_decisions": decisions,
    }


def _scores(
    prediction: np.ndarray,
    observed: np.ndarray,
    arrays: dict[str, np.ndarray],
    layer_index: int,
    mask: np.ndarray,
) -> dict[str, float]:
    centered_prediction = prediction - prediction.mean(axis=1, keepdims=True)
    centered_observed = observed - observed.mean(axis=1, keepdims=True)
    power = max(float(np.mean(centered_observed**2)), 1e-12)
    predicted_candidate = (
        arrays["clean_answer_logits"][arrays["recipient_id"][mask]] + prediction
    ).argmax(axis=1)
    direct_candidate = arrays["intervened_answer_logits"][layer_index, mask].argmax(axis=1)
    donors = arrays["donor_id"][mask]
    return {
        "contrast_normalized_mse": float(
            np.mean((centered_prediction - centered_observed) ** 2) / power
        ),
        "contrast_correlation": effect_correlation(centered_prediction, centered_observed),
        "answer_candidate_agreement": float(np.mean(predicted_candidate == direct_candidate)),
        "predicted_donor_candidate_rate": float(np.mean(predicted_candidate == donors)),
        "direct_donor_candidate_rate": float(np.mean(direct_candidate == donors)),
    }


def _transition_decision(config: dict[str, Any], behavior: dict[str, Any]) -> bool:
    early_max = float(config.get("early_donor_transfer_max", 0.10))
    late_min = float(config.get("late_donor_transfer_min", 0.60))
    increase = float(config.get("donor_transfer_increase_min", 0.50))
    for split in ("validation", "test"):
        early = max(
            behavior["18"][split]["full_vocab_donor_token_transfer"],
            behavior["21"][split]["full_vocab_donor_token_transfer"],
        )
        late = min(
            behavior["24"][split]["full_vocab_donor_token_transfer"],
            behavior["26"][split]["full_vocab_donor_token_transfer"],
        )
        if not (early <= early_max and late >= late_min and late - early >= increase):
            return False
    return True


def _terminal_transition_decision(config: dict[str, Any], behavior: dict[str, Any]) -> bool:
    early_max = float(config.get("early_donor_transfer_max", 0.10))
    terminal_min = float(config.get("terminal_donor_transfer_min", 0.60))
    increase = float(config.get("donor_transfer_increase_min", 0.50))
    for split in ("validation", "test"):
        early = max(
            behavior["18"][split]["full_vocab_donor_token_transfer"],
            behavior["21"][split]["full_vocab_donor_token_transfer"],
        )
        terminal = behavior["26"][split]["full_vocab_donor_token_transfer"]
        if not (
            early <= early_max
            and terminal >= terminal_min
            and terminal - early >= increase
        ):
            return False
    return True


def _inversion_decision(config: dict[str, Any], scores: dict[str, Any]) -> bool:
    early_ratio = float(config.get("early_local_to_population_ratio_max", 0.25))
    late_ratio = float(config.get("late_population_to_local_ratio_max", 0.60))
    local_jump = float(config.get("local_mse_jump_min", 3.0))
    correlation_margin = float(config.get("late_population_correlation_margin_min", 0.01))
    for split in ("validation", "test"):
        early = scores["21"]["splits"][split]
        late = scores["24"]["splits"][split]
        early_local = early["local_jacobian"]["contrast_normalized_mse"]
        early_population = early["train_population_jacobian"]["contrast_normalized_mse"]
        late_local = late["local_jacobian"]["contrast_normalized_mse"]
        late_population = late["train_population_jacobian"]["contrast_normalized_mse"]
        if not (
            early_local <= early_population * early_ratio
            and late_population <= late_local * late_ratio
            and late_local >= early_local * local_jump
            and late["train_population_jacobian"]["contrast_correlation"]
            >= late["local_jacobian"]["contrast_correlation"] + correlation_margin
        ):
            return False
    return True


def _semantic_decision(config: dict[str, Any], scores: dict[str, Any]) -> bool:
    for layer in ("24", "26"):
        real = scores[layer]["splits"]["test"]["train_population_jacobian"]
        null = scores[layer]["row_permutation_null_test"]
        if not (
            real["contrast_normalized_mse"]
            <= null["p05_contrast_normalized_mse"]
            * float(config.get("row_null_mse_ratio_max", 0.80))
            and real["answer_candidate_agreement"]
            >= null["p95_answer_candidate_agreement"]
            + float(config.get("row_null_candidate_margin_min", 0.05))
        ):
            return False
    return True


def _population_advantage_details(
    scores: dict[str, Any], behavior: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("validation", "test"):
        donor_control = []
        advantages = []
        by_layer = {}
        for layer in ("18", "21", "24", "26"):
            layer_scores = scores[layer]["splits"][split]
            local_mse = layer_scores["local_jacobian"]["contrast_normalized_mse"]
            quadratic_mse = layer_scores["quadratic_taylor"]["contrast_normalized_mse"]
            population_mse = layer_scores["train_population_jacobian"][
                "contrast_normalized_mse"
            ]
            best_local = min(local_mse, quadratic_mse)
            advantage = best_local - population_mse
            control = behavior[layer][split]["full_vocab_donor_token_transfer"]
            donor_control.append(control)
            advantages.append(advantage)
            by_layer[layer] = {
                "best_local_or_hvp_mse": best_local,
                "population_mse": population_mse,
                "population_advantage": advantage,
                "full_vocab_donor_transfer": control,
            }
        result[split] = {
            "by_layer": by_layer,
            "donor_control_advantage_spearman": _spearman(donor_control, advantages),
        }
    return result


def _control_population_association_decision(
    config: dict[str, Any], details: dict[str, Any]
) -> bool:
    early_margin = float(config.get("early_best_transport_margin_min", 0.05))
    late_margin = float(config.get("late_population_advantage_margin_min", 0.0))
    correlation_min = float(config.get("control_advantage_spearman_min", 0.80))
    for split in ("validation", "test"):
        split_details = details[split]
        early_advantage = split_details["by_layer"]["21"]["population_advantage"]
        late_advantage = split_details["by_layer"]["24"]["population_advantage"]
        if not (
            early_advantage <= -early_margin
            and late_advantage >= late_margin
            and split_details["donor_control_advantage_spearman"] >= correlation_min
        ):
            return False
    return True


def _boundary_alignment_details(
    config: dict[str, Any], details: dict[str, Any]
) -> dict[str, Any]:
    control_threshold = float(config.get("control_boundary_transfer_min", 0.50))
    advantage_threshold = float(config.get("advantage_boundary_min", 0.0))
    result = {}
    for split in ("validation", "test"):
        by_layer = details[split]["by_layer"]
        control_layer = next(
            (
                int(layer)
                for layer in ("18", "21", "24", "26")
                if by_layer[layer]["full_vocab_donor_transfer"] >= control_threshold
            ),
            None,
        )
        advantage_layer = next(
            (
                int(layer)
                for layer in ("18", "21", "24", "26")
                if by_layer[layer]["population_advantage"] >= advantage_threshold
            ),
            None,
        )
        result[split] = {
            "control_boundary_layer": control_layer,
            "population_advantage_boundary_layer": advantage_layer,
            "boundaries_equal": bool(
                control_layer is not None and control_layer == advantage_layer
            ),
        }
    return result


def _boundary_alignment_decision(
    config: dict[str, Any],
    details: dict[str, Any],
    boundaries: dict[str, Any],
) -> bool:
    correlation_min = float(config.get("control_advantage_spearman_min", 0.80))
    if not _boundary_sign_equality_decision(config, details, boundaries):
        return False
    for split in ("validation", "test"):
        if details[split]["donor_control_advantage_spearman"] < correlation_min:
            return False
    return True


def _boundary_sign_equality_decision(
    config: dict[str, Any],
    details: dict[str, Any],
    boundaries: dict[str, Any],
) -> bool:
    early_margin = float(config.get("early_best_transport_margin_min", 0.05))
    terminal_margin = float(config.get("terminal_population_advantage_min", 0.05))
    for split in ("validation", "test"):
        split_details = details[split]
        if not (
            split_details["by_layer"]["21"]["population_advantage"] <= -early_margin
            and split_details["by_layer"]["26"]["population_advantage"]
            >= terminal_margin
            and boundaries[split]["boundaries_equal"]
        ):
            return False
    return True


def _boundary_semantic_decision(
    config: dict[str, Any],
    scores: dict[str, Any],
    boundaries: dict[str, Any],
) -> bool:
    boundary = boundaries["test"]["control_boundary_layer"]
    if boundary is None:
        return False
    for layer in {str(boundary), "26"}:
        real = scores[layer]["splits"]["test"]["train_population_jacobian"]
        null = scores[layer]["row_permutation_null_test"]
        if not (
            real["contrast_normalized_mse"]
            <= null["p05_contrast_normalized_mse"]
            * float(config.get("row_null_mse_ratio_max", 0.80))
            and real["answer_candidate_agreement"]
            >= null["p95_answer_candidate_agreement"]
            + float(config.get("row_null_candidate_margin_min", 0.05))
        ):
            return False
    return True


def _spearman(left: list[float], right: list[float]) -> float:
    left_rank = _rankdata(np.asarray(left, dtype=np.float64))
    right_rank = _rankdata(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _row_permutation_null(
    config: dict[str, Any],
    population_jacobian: np.ndarray,
    arrays: dict[str, np.ndarray],
    layer_index: int,
    observed: np.ndarray,
    *,
    split: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    mask = arrays["split_id"] == split
    values = []
    for _ in range(int(config.get("row_permutations", 256))):
        permuted = population_jacobian[rng.permutation(population_jacobian.shape[0])]
        prediction = arrays["source_delta"][layer_index, mask] @ permuted.T
        values.append(_scores(prediction, observed[mask], arrays, layer_index, mask))
    mse = np.asarray([value["contrast_normalized_mse"] for value in values])
    agreement = np.asarray([value["answer_candidate_agreement"] for value in values])
    return {
        "permutations": len(values),
        "p05_contrast_normalized_mse": float(np.quantile(mse, 0.05)),
        "mean_contrast_normalized_mse": float(mse.mean()),
        "p95_contrast_normalized_mse": float(np.quantile(mse, 0.95)),
        "p05_answer_candidate_agreement": float(np.quantile(agreement, 0.05)),
        "mean_answer_candidate_agreement": float(agreement.mean()),
        "p95_answer_candidate_agreement": float(np.quantile(agreement, 0.95)),
    }


def _averaging_curve(
    config: dict[str, Any],
    jacobians: np.ndarray,
    train_entities: np.ndarray,
    arrays: dict[str, np.ndarray],
    layer_index: int,
    observed: np.ndarray,
    *,
    split: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mask = arrays["split_id"] == split
    result = {}
    draws = int(config.get("averaging_draws", 128))
    for size in (1, 2, 4, 8, 16, 24):
        actual = 1 if size == len(train_entities) else draws
        values = []
        for _ in range(actual):
            chosen = rng.choice(train_entities, size=size, replace=False)
            prediction = arrays["source_delta"][layer_index, mask] @ jacobians[chosen].mean(0).T
            values.append(_scores(prediction, observed[mask], arrays, layer_index, mask))
        result[str(size)] = {
            "draws": actual,
            "median_contrast_normalized_mse": float(
                np.median([value["contrast_normalized_mse"] for value in values])
            ),
            "median_answer_candidate_agreement": float(
                np.median([value["answer_candidate_agreement"] for value in values])
            ),
        }
    return result


def _split_name(split: int) -> str:
    return ("train", "validation", "test")[split]


def _numpy(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype(np.float32)


def _config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


__all__ = [
    "ELEMENT_PAIRS",
    "ELEMENT_SPLIT_IDS",
    "STATE_PAIRS",
    "STATE_SPLIT_IDS",
    "_boundary_alignment_decision",
    "_boundary_alignment_details",
    "_boundary_sign_equality_decision",
    "_boundary_semantic_decision",
    "_control_population_association_decision",
    "_inversion_decision",
    "_population_advantage_details",
    "_relation_spec",
    "_scores",
    "_spearman",
    "_transition_decision",
    "_terminal_transition_decision",
    "_within_split_pairs",
    "run_qwen_element_layer_geometry_study",
]
