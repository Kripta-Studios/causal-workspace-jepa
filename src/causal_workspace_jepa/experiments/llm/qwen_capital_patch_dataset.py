"""Behavior-changing Qwen country-capital residual-patch dataset."""

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
from causal_workspace_jepa.experiments.llm.qwen_jvp_audit import (
    _directional_output_function,
    _float32_endpoint_tolerance,
    _target_vector,
)
from causal_workspace_jepa.hooks.names import transformer_site


CAPITAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Portugal", "Lisbon"),
    ("Greece", "Athens"),
    ("Norway", "Oslo"),
    ("Sweden", "Stockholm"),
    ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"),
    ("Poland", "Warsaw"),
    ("Austria", "Vienna"),
    ("Hungary", "Budapest"),
    ("Bulgaria", "Sofia"),
    ("Ireland", "Dublin"),
    ("Belgium", "Brussels"),
    ("Netherlands", "Amsterdam"),
    ("Switzerland", "Bern"),
    ("Egypt", "Cairo"),
    ("Turkey", "Ankara"),
    ("Thailand", "Bangkok"),
    ("Indonesia", "Jakarta"),
    ("Australia", "Canberra"),
    ("Chile", "Santiago"),
    ("Peru", "Lima"),
    ("Cuba", "Havana"),
    ("Tunisia", "Tunis"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    ("Pakistan", "Islamabad"),
    ("Russia", "Moscow"),
    ("Philippines", "Manila"),
    ("South Korea", "Seoul"),
    ("North Korea", "Pyongyang"),
    ("Singapore", "Singapore"),
    ("New Zealand", "Wellington"),
)

SPLIT_COUNTRIES: dict[str, tuple[str, ...]] = {
    "train": (
        "Iraq",
        "Netherlands",
        "Italy",
        "Egypt",
        "Indonesia",
        "Philippines",
        "Sweden",
        "Bulgaria",
        "Iran",
        "Spain",
        "Chile",
        "Belgium",
        "Greece",
        "Pakistan",
        "South Korea",
        "Peru",
        "Ireland",
        "Turkey",
        "France",
        "Poland",
        "Switzerland",
        "Portugal",
        "Russia",
        "Singapore",
    ),
    "validation": ("Hungary", "Norway", "Cuba", "Australia", "North Korea", "Denmark"),
    "test": ("Thailand", "Finland", "New Zealand", "Tunisia", "Germany", "Austria"),
}


def run_qwen_capital_patch_dataset(config_path: str | Path) -> dict[str, Any]:
    """Generate direct, JVP, and quadratic effects for entity-disjoint donor patches."""

    import torch

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    hardware = require_free_disk(
        str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    )
    seed = int(config.get("seed", 223))
    provenance = collect_provenance(
        command=f"python scripts/generate_qwen_capital_patches.py --config {config_path.as_posix()}",
        resource_profile=str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml")),
        seed=seed,
    )
    if provenance.git_dirty:
        raise RuntimeError("LLM-CAPITAL-PATCH-001 requires a clean committed worktree")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

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
    hidden_size = int(adapter.model.config.hidden_size)
    source_layer = int(config.get("source_layer", 21))
    source_site = transformer_site(source_layer, "resid_post")
    target_site = transformer_site(len(adapter.layers) - 1, "resid_post")
    prompts = [f"The capital of {country} is" for country, _capital in CAPITAL_PAIRS]
    answer_ids = _single_token_answer_ids(adapter)
    answer_tensor = torch.as_tensor(answer_ids, device=adapter.device)
    identity = torch.eye(hidden_size, device=adapter.device)
    expected = sum(
        len(countries) * (len(countries) - 1) for countries in SPLIT_COUNTRIES.values()
    )
    estimated_bytes = int(expected * (9 * hidden_size + 6 * len(answer_ids) + 16) * 4 * 1.25)
    budget_bytes = int(float(config.get("activation_budget_mb", 64)) * 1024**2)
    if estimated_bytes > budget_bytes:
        raise RuntimeError(
            f"BLOCKED_RESOURCE: estimated capital patch data {estimated_bytes} "
            f"exceeds {budget_bytes}"
        )
    with torch.no_grad():
        clean_runs = [
            adapter.forward_with_cache(
                adapter.tokenize([prompt]), [source_site, target_site, "logits"]
            )
            for prompt in prompts
        ]
    for entity_id, run in enumerate(clean_runs):
        adapter.register_donor(f"capital_{entity_id}", source_site, run.activations[source_site])

    arrays: dict[str, list[np.ndarray | int | float]] = {
        "clean_source": [],
        "donor_source": [],
        "source_delta": [],
        "clean_target_hidden": [],
        "intervened_target_hidden": [],
        "clean_answer_logits": [],
        "intervened_answer_logits": [],
        "target_effect": [],
        "exact_jvp": [],
        "quadratic_taylor": [],
        "central_difference": [],
        "recipient_id": [],
        "donor_id": [],
        "split_id": [],
        "recipient_answer_id": [],
        "donor_answer_id": [],
        "clean_top_token": [],
        "intervened_top_token": [],
        "direct_source_semantic_error": [],
        "source_endpoint_error": [],
        "source_endpoint_tolerance": [],
        "jvp_central_relative_error": [],
    }
    records: list[dict[str, Any]] = []
    central_epsilon = float(config.get("central_epsilon", 0.125))
    country_to_id = {country: index for index, (country, _capital) in enumerate(CAPITAL_PAIRS)}
    split_names = ("train", "validation", "test")
    for split_id, split_name in enumerate(split_names):
        entity_ids = [country_to_id[country] for country in SPLIT_COUNTRIES[split_name]]
        for recipient_id in entity_ids:
            clean = clean_runs[recipient_id]
            clean_source = clean.activations[source_site][0, -1].detach().float()
            clean_hidden = clean.activations[target_site][0, -1].detach().float()
            clean_logits = clean.logits[0, -1].detach().float()
            clean_vector = _target_vector(clean, target_site, identity, answer_tensor).detach()
            for donor_id in entity_ids:
                if donor_id == recipient_id:
                    continue
                donor_source = clean_runs[donor_id].activations[source_site][0, -1].detach().float()
                direction = donor_source - clean_source
                spec = InterventionSpec(
                    site=source_site,
                    operation="patch",
                    positions=(-1,),
                    donor_example_id=f"capital_{donor_id}",
                    seed=seed,
                )
                with torch.no_grad():
                    direct = adapter.forward_with_intervention(
                        clean.token_batch, spec, [source_site, target_site, "logits"]
                    )
                    direct_vector = _target_vector(
                        direct, target_site, identity, answer_tensor
                    ).detach()
                function = _directional_output_function(
                    adapter,
                    clean.token_batch,
                    source_site=source_site,
                    target_site=target_site,
                    direction=direction,
                    hidden_projection=identity,
                    logit_ids=answer_tensor,
                )
                scale = torch.zeros((), device=adapter.device, requires_grad=True)
                _, jvp = torch.autograd.functional.jvp(
                    function,
                    scale,
                    torch.ones_like(scale),
                    create_graph=False,
                    strict=True,
                )
                with torch.no_grad():
                    plus = function(torch.tensor(central_epsilon, device=adapter.device))
                    minus = function(torch.tensor(-central_epsilon, device=adapter.device))
                central = (plus - minus) / (2.0 * central_epsilon)
                second = (plus - 2.0 * clean_vector + minus) / (central_epsilon**2)
                quadratic = jvp + 0.5 * second
                effect = direct_vector - clean_vector
                direct_source = direct.activations[source_site][0, -1].detach().float()
                endpoint = clean_source + direction
                relative = float(
                    (central - jvp).norm().detach().cpu()
                    / max(float(jvp.norm().detach().cpu()), 1e-8)
                )
                arrays["clean_source"].append(_numpy(clean_source))
                arrays["donor_source"].append(_numpy(donor_source))
                arrays["source_delta"].append(_numpy(direction))
                arrays["clean_target_hidden"].append(_numpy(clean_hidden))
                arrays["intervened_target_hidden"].append(
                    _numpy(direct.activations[target_site][0, -1])
                )
                arrays["clean_answer_logits"].append(_numpy(clean_logits[answer_tensor]))
                arrays["intervened_answer_logits"].append(
                    _numpy(direct.logits[0, -1, answer_tensor])
                )
                arrays["target_effect"].append(_numpy(effect))
                arrays["exact_jvp"].append(_numpy(jvp))
                arrays["quadratic_taylor"].append(_numpy(quadratic))
                arrays["central_difference"].append(_numpy(central))
                arrays["recipient_id"].append(recipient_id)
                arrays["donor_id"].append(donor_id)
                arrays["split_id"].append(split_id)
                arrays["recipient_answer_id"].append(answer_ids[recipient_id])
                arrays["donor_answer_id"].append(answer_ids[donor_id])
                arrays["clean_top_token"].append(int(clean_logits.argmax().cpu()))
                arrays["intervened_top_token"].append(
                    int(direct.logits[0, -1].argmax().detach().cpu())
                )
                arrays["direct_source_semantic_error"].append(
                    float((direct_source - donor_source).abs().max().cpu())
                )
                arrays["source_endpoint_error"].append(
                    float((endpoint - donor_source).abs().max().cpu())
                )
                arrays["source_endpoint_tolerance"].append(
                    _float32_endpoint_tolerance(clean_source, donor_source)
                )
                arrays["jvp_central_relative_error"].append(relative)
                recipient_country, recipient_capital = CAPITAL_PAIRS[recipient_id]
                donor_country, donor_capital = CAPITAL_PAIRS[donor_id]
                records.append(
                    {
                        "example_id": f"capital-{split_name}-{recipient_id:02d}-{donor_id:02d}",
                        "split": split_name,
                        "recipient_id": recipient_id,
                        "recipient_country": recipient_country,
                        "recipient_capital": recipient_capital,
                        "donor_id": donor_id,
                        "donor_country": donor_country,
                        "donor_capital": donor_capital,
                        "source_site": source_site,
                        "target_site": target_site,
                        "intervention": spec.to_dict(),
                    }
                )

    converted = {name: np.asarray(values, dtype=_dtype(name)) for name, values in arrays.items()}
    if len(records) != expected:
        raise RuntimeError(f"generated {len(records)} capital patches, expected {expected}")
    storage = write_hdf5_shards(
        str(config.get("output_dir", "data/activations/qwen_capital_patches_v1")),
        converted,
        records,
        dataset_id=str(config.get("id", "LLM-CAPITAL-PATCH-001")),
        config_digest=_config_digest(config),
        max_shard_mb=float(config.get("max_shard_mb", 32)),
        budget_mb=float(config.get("activation_budget_mb", 64)),
        resume=bool(config.get("resume", True)),
    )
    metrics = _capital_metrics(config, converted, answer_ids)
    metrics.update(
        {
            "experiment_id": str(config.get("id", "LLM-CAPITAL-PATCH-001")),
            "status": "SMOKE_VALIDATED" if metrics["numerical_audit_passed"] else "REJECTED",
            "evidence_level": "Causal mediation",
            "model": str(config.get("model")),
            "model_revision": adapter._metadata()["resolved_revision"],
            "precision": "float32",
            "source_site": source_site,
            "target_site": target_site,
            "entities": len(CAPITAL_PAIRS),
            "outcomes": expected,
            "estimated_bytes_before_capture": estimated_bytes,
            "answer_token_ids": answer_ids,
            "splits": {name: list(countries) for name, countries in SPLIT_COUNTRIES.items()},
            "storage": storage,
            "hardware": hardware.as_dict(),
            "runtime_seconds": float(time.perf_counter() - started),
            "scientific_boundary": (
                "This dataset establishes direct donor-patch behavior and local baselines only. "
                "It does not validate a meta-model, feature meaning, circuit, or workspace."
            ),
        }
    )
    output_metrics = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen_capital_patch_dataset_v1.json"))
    )
    output_manifest = Path(
        str(config.get("output_manifest", "data/manifests/qwen_capital_patches_v1.json"))
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
        "target_site": target_site,
        "local_data_root": str(config.get("output_dir")),
        "entity_split": {name: list(countries) for name, countries in SPLIT_COUNTRIES.items()},
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
            "behavior_eligible": metrics["behavior_eligible"],
        },
    )
    if not metrics["numerical_audit_passed"]:
        raise RuntimeError("capital patch dataset failed numerical validity gates")
    return metrics


def capital_split_ids() -> np.ndarray:
    """Return the immutable entity-level split assignment in CAPITAL_PAIRS order."""

    assignment: dict[str, int] = {}
    for split_id, split_name in enumerate(("train", "validation", "test")):
        assignment.update({country: split_id for country in SPLIT_COUNTRIES[split_name]})
    return np.asarray([assignment[country] for country, _capital in CAPITAL_PAIRS], dtype=np.int64)


def _single_token_answer_ids(adapter: QwenHFAdapter) -> list[int]:
    answer_ids: list[int] = []
    for _country, capital in CAPITAL_PAIRS:
        token_ids = adapter.tokenizer.encode(f" {capital}", add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(f"preregistered answer is not one token: {capital} -> {token_ids}")
        answer_ids.append(int(token_ids[0]))
    if len(set(answer_ids)) != len(answer_ids):
        raise RuntimeError("capital answer token IDs must be unique")
    return answer_ids


def _capital_metrics(
    config: dict[str, Any], arrays: dict[str, np.ndarray], answer_ids: list[int]
) -> dict[str, Any]:
    split = arrays["split_id"]
    clean_correct = arrays["clean_top_token"] == arrays["recipient_answer_id"]
    donor_transfer = arrays["intervened_top_token"] == arrays["donor_answer_id"]
    top_changed = arrays["intervened_top_token"] != arrays["clean_top_token"]
    effect = arrays["target_effect"]
    jvp = arrays["exact_jvp"]
    quadratic = arrays["quadratic_taylor"]
    numerical_gates = {
        "direct_source_semantics": float(arrays["direct_source_semantic_error"].max()) == 0.0,
        "direction_endpoint_roundoff": bool(
            np.all(arrays["source_endpoint_error"] <= arrays["source_endpoint_tolerance"])
        ),
        "median_jvp_central_relative_error": float(
            np.median(arrays["jvp_central_relative_error"])
        )
        <= float(config.get("median_relative_error_max", 0.05)),
        "p95_jvp_central_relative_error": float(
            np.quantile(arrays["jvp_central_relative_error"], 0.95)
        )
        <= float(config.get("p95_relative_error_max", 0.15)),
        "all_effects_nonzero": bool(np.all(np.linalg.norm(effect, axis=1) > 0)),
    }
    clean_by_split = {
        name: float(clean_correct[split == split_id].mean())
        for split_id, name in enumerate(("train", "validation", "test"))
    }
    transfer_by_split = {
        name: float(donor_transfer[split == split_id].mean())
        for split_id, name in enumerate(("train", "validation", "test"))
    }
    behavior_gates = {
        "clean_accuracy_each_split": min(clean_by_split.values())
        >= float(config.get("clean_accuracy_min", 0.60)),
        "test_donor_transfer": transfer_by_split["test"]
        >= float(config.get("test_donor_transfer_min", 0.40)),
        "overall_top_token_change": float(top_changed.mean())
        >= float(config.get("top_token_change_min", 0.40)),
    }
    answer_count = len(answer_ids)
    direct_candidate = arrays["intervened_answer_logits"].argmax(axis=1)
    jvp_candidate = (arrays["clean_answer_logits"] + jvp[:, -answer_count:]).argmax(axis=1)
    quadratic_candidate = (
        arrays["clean_answer_logits"] + quadratic[:, -answer_count:]
    ).argmax(axis=1)
    return {
        "numerical_audit_passed": bool(all(numerical_gates.values())),
        "behavior_eligible": bool(all(behavior_gates.values())),
        "numerical_gates": numerical_gates,
        "behavior_gates": behavior_gates,
        "clean_top_token_accuracy_by_split": clean_by_split,
        "donor_top_token_transfer_by_split": transfer_by_split,
        "overall_top_token_change_rate": float(top_changed.mean()),
        "exact_jvp_normalized_mse": _normalized_mse(jvp, effect),
        "quadratic_normalized_mse": _normalized_mse(quadratic, effect),
        "exact_jvp_candidate_agreement": float(np.mean(jvp_candidate == direct_candidate)),
        "quadratic_candidate_agreement": float(
            np.mean(quadratic_candidate == direct_candidate)
        ),
        "median_jvp_central_relative_error": float(
            np.median(arrays["jvp_central_relative_error"])
        ),
        "p95_jvp_central_relative_error": float(
            np.quantile(arrays["jvp_central_relative_error"], 0.95)
        ),
        "direct_source_semantic_max_abs": float(
            arrays["direct_source_semantic_error"].max()
        ),
        "source_endpoint_tolerance_violations": int(
            np.sum(arrays["source_endpoint_error"] > arrays["source_endpoint_tolerance"])
        ),
    }


def _normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2) / max(float(np.mean(target**2)), 1e-12))


def _numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().astype(np.float32)


def _dtype(name: str) -> np.dtype[Any]:
    if name.endswith("_id") or name.endswith("_token"):
        return np.dtype(np.int64)
    return np.dtype(np.float32)


def _config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
