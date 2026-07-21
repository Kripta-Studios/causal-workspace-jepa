"""Budgeted, resumable selected-site Qwen activation capture."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from causal_workspace_jepa.adapters.qwen_hf_adapter import QwenAdapterConfig, QwenHFAdapter
from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import estimate_activation_bytes, require_free_disk
from causal_workspace_jepa.data.activation_store import write_hdf5_shards
from causal_workspace_jepa.experiments.llm.qwen_intervention_dataset import qwen_intervention_prompts
from causal_workspace_jepa.hooks.names import transformer_site


POSITION_KINDS = {"last": 0, "filler": 1, "answer": 2}


def run_qwen_activation_capture(config_path: str | Path) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    resource = require_free_disk(resource_profile)
    seed = int(config.get("seed", 113))
    torch.manual_seed(seed)
    np.random.seed(seed)
    provenance = collect_provenance(
        command=f"python scripts/capture_qwen_activations.py --config {config_path}",
        resource_profile=resource_profile,
        seed=seed,
    )
    if provenance.git_dirty:
        raise RuntimeError("activation capture requires a clean committed worktree")
    prompts, split_ids, _families = qwen_intervention_prompts()
    prompt_count = min(int(config.get("prompt_count", len(prompts))), len(prompts))
    prompts = prompts[:prompt_count]
    split_ids = split_ids[:prompt_count]
    layers = tuple(int(layer) for layer in config.get("selected_layers", [0, 8, 16, 24, 31]))
    position_kinds = tuple(str(kind) for kind in config.get("selected_positions", ["last"]))
    unknown_positions = set(position_kinds).difference(POSITION_KINDS)
    if unknown_positions:
        raise ValueError(f"unknown selected position kinds: {sorted(unknown_positions)}")
    adapter = QwenHFAdapter(
        QwenAdapterConfig(
            model_name=str(config.get("model", "Qwen/Qwen3-4B")),
            revision=str(config["revision"]),
            device=str(config.get("device", "cuda")),
            dtype=str(config.get("dtype", "bfloat16")),
            max_length=int(config.get("max_sequence_length", 128)),
            local_files_only=bool(config.get("local_files_only", False)),
            token=False,
        )
    )
    if max(layers) >= len(adapter.layers) or min(layers) < 0:
        raise ValueError("selected layer is outside the loaded Qwen model")
    hidden_size = int(adapter.model.config.hidden_size)
    estimate = estimate_activation_bytes(
        examples=prompt_count,
        layers=len(layers),
        positions=len(position_kinds),
        hidden_size=hidden_size,
        bytes_per_value=2,
        overhead_fraction=0.5,
    )
    budget = int(float(config.get("activation_budget_mb", 64)) * 1024**2)
    if estimate > budget:
        raise RuntimeError(
            f"BLOCKED_RESOURCE: estimated activation capture {estimate} exceeds {budget}"
        )
    sites = [transformer_site(layer, "resid_post") for layer in layers]
    arrays: dict[str, list[Any]] = {
        "activation": [],
        "prompt_id": [],
        "split_id": [],
        "layer": [],
        "position": [],
        "position_kind": [],
        "token_id": [],
        "top_logit_id": [],
        "top_logit_value": [],
    }
    for prompt_id, prompt in enumerate(prompts):
        batch = adapter.tokenize([prompt])
        run = adapter.forward_with_cache(batch, [*sites, "logits"])
        length = int(run.token_batch.attention_mask[0].detach().cpu().sum().item())
        positions = resolve_selected_positions(length, position_kinds)
        logits = np.asarray(run.logits)[0]
        token_ids = np.asarray(run.token_batch.input_ids.detach().cpu())[0]
        for layer, site in zip(layers, sites, strict=True):
            activation = np.asarray(run.activations[site])[0]
            for kind, position in zip(position_kinds, positions, strict=True):
                row_logits = logits[position]
                top_id = int(np.argmax(row_logits))
                arrays["activation"].append(activation[position].astype(np.float16))
                arrays["prompt_id"].append(prompt_id)
                arrays["split_id"].append(int(split_ids[prompt_id]))
                arrays["layer"].append(layer)
                arrays["position"].append(position)
                arrays["position_kind"].append(POSITION_KINDS[kind])
                arrays["token_id"].append(int(token_ids[position]))
                arrays["top_logit_id"].append(top_id)
                arrays["top_logit_value"].append(float(row_logits[top_id]))
    packed = {name: np.asarray(values) for name, values in arrays.items()}
    output_dir = Path(str(config.get("output_dir", "data/activations/qwen_selected_layers")))
    config_identity = {
        "model": str(config.get("model")),
        "revision": str(config.get("revision")),
        "layers": list(layers),
        "position_kinds": list(position_kinds),
        "prompt_count": prompt_count,
        "seed": seed,
    }
    config_digest = hashlib.sha256(
        json.dumps(config_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    records = [
        {
            "prompt_id": int(packed["prompt_id"][index]),
            "split_id": int(packed["split_id"][index]),
            "layer": int(packed["layer"][index]),
            "position": int(packed["position"][index]),
            "position_kind": int(packed["position_kind"][index]),
        }
        for index in range(len(packed["prompt_id"]))
    ]
    manifest = write_hdf5_shards(
        output_dir,
        packed,
        records,
        dataset_id=str(config.get("id", "qwen_selected_layers")),
        config_digest=config_digest,
        max_shard_mb=float(config.get("shard_size_mb", 32)),
        budget_mb=float(config.get("activation_budget_mb", 64)),
    )
    committed_manifest_path = Path(
        str(
            config.get(
                "committed_manifest",
                "data/manifests/qwen_selected_activation_capture.json",
            )
        )
    )
    committed_manifest = {
        **manifest,
        "model": str(config.get("model")),
        "revision": str(config.get("revision")),
        "source_prompts": "repository-authored fixed LLM-INTDATA-001 prompts",
        "license": "generated",
        "split": "fixed 8/2/2 prompt split",
        "activation_root": output_dir.as_posix(),
        "shards": [
            {**shard, "path": (output_dir / str(shard["path"])).as_posix()}
            for shard in manifest["shards"]
        ],
    }
    committed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    committed_manifest_path.write_text(
        json.dumps(committed_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved_revision = adapter._metadata()["resolved_revision"]
    passes = {
        "requested_revision_resolved": resolved_revision == str(config.get("revision")),
        "row_count": len(packed["prompt_id"])
        == prompt_count * len(layers) * len(position_kinds),
        "budget_respected": int(manifest["estimated_uncompressed_bytes"]) <= budget,
        "checksummed_shards": bool(manifest["shards"])
        and all(len(str(shard["sha256"])) == 64 for shard in manifest["shards"]),
    }
    all_passed = all(passes.values())
    metrics: dict[str, Any] = {
        "experiment_id": "LLM-QWEN-CAPTURE-001",
        "status": "SMOKE_VALIDATED" if all_passed else "FAILED_REGISTERED_GATES",
        "evidence_level": "Availability",
        "all_passed": all_passed,
        "passes": passes,
        "model": str(config.get("model")),
        "requested_revision": str(config.get("revision")),
        "resolved_revision": resolved_revision,
        "rows": int(len(packed["prompt_id"])),
        "prompts": prompt_count,
        "layers": list(layers),
        "position_kinds": list(position_kinds),
        "hidden_size": hidden_size,
        "estimated_bytes": estimate,
        "storage": manifest,
        "committed_manifest": committed_manifest_path.as_posix(),
        "hardware": resource.as_dict(),
        "runtime_seconds": float(time.monotonic() - started),
    }
    output_metrics = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen_selected_activation_capture.json"))
    )
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output_metrics.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": output_metrics.as_posix()},
    )
    if not all_passed:
        failed = [name for name, passed in passes.items() if not passed]
        raise RuntimeError(f"Qwen activation capture failed registered gates: {failed}")
    return metrics


def resolve_selected_positions(length: int, kinds: Sequence[str]) -> tuple[int, ...]:
    """Resolve semantic position labels without scanning or saving all tokens."""

    if length < 1:
        raise ValueError("tokenized sequence must be nonempty")
    mapping = {
        "last": length - 1,
        "answer": length - 1,
        "filler": max(0, (length - 1) // 2),
    }
    return tuple(mapping[kind] for kind in kinds)
