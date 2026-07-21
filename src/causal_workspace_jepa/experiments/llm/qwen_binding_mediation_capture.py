"""Capture the preregistered Qwen binding treatment trajectories.

The capture is intentionally separate from candidate ranking and mediation
analysis.  It executes the frozen upstream treatment, checks exact donor replay,
and stores only the selected module/query trajectories needed by the later
analysis.  Per-episode progress files make the long GPU run resumable without
pretending a partial dataset is complete.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal_workspace_jepa.adapters.qwen_hf_adapter import QwenAdapterConfig, QwenHFAdapter
from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.data.activation_store import read_hdf5_shards, write_hdf5_shards
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_protocol import (
    BindingEpisode,
    TokenizedTreatment,
    audit_token_pools,
    audit_tokenized_treatment,
    binding_episodes_from_config,
    tokenization_digest,
    tokenized_treatment_payload,
)
from causal_workspace_jepa.hooks.names import transformer_site


def run_qwen_binding_mediation_capture(config_path: str | Path) -> dict[str, Any]:
    """Execute and store all frozen clean/donor/treatment trajectories."""

    import torch

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config["resource_profile"])
    hardware = require_free_disk(resource_profile)
    seed = int(config["seed"])
    provenance = collect_provenance(
        command=(
            "python scripts/capture_qwen_binding_mediation.py "
            f"--config {config_path.as_posix()}"
        ),
        resource_profile=resource_profile,
        seed=seed,
    )
    if provenance.git_dirty:
        raise RuntimeError("Qwen binding capture requires a clean committed worktree")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    adapter = QwenHFAdapter(
        QwenAdapterConfig(
            model_name=str(config["model"]),
            revision=str(config["revision"]),
            device=str(config["device"]),
            dtype=str(config["dtype"]),
            max_length=int(config["max_sequence_length"]),
            local_files_only=bool(config["local_files_only"]),
            preserve_autograd=False,
            attn_implementation=str(config["attn_implementation"]),
            token=False,
        )
    )
    adapter.model.eval()
    model_metadata = dict(adapter._metadata())
    resolved_revision = model_metadata["resolved_revision"]
    if resolved_revision != str(config["revision"]):
        raise RuntimeError(
            "resolved Qwen revision differs from the frozen revision: "
            f"{resolved_revision!r} != {config['revision']!r}"
        )
    if len(adapter.layers) != 28:
        raise RuntimeError(f"registered candidate count assumes 28 layers, got {len(adapter.layers)}")
    candidate_sites = tuple(
        transformer_site(layer, kind)
        for layer in range(len(adapter.layers))
        for kind in ("attn_out", "mlp_out")
    )
    if len(candidate_sites) != int(config["candidates"]["node_count"]):
        raise RuntimeError("candidate site count differs from the frozen configuration")
    treatment_site = str(config["treatment"]["site"])
    final_site = transformer_site(len(adapter.layers) - 1, "resid_post")
    episodes = binding_episodes_from_config(config)
    token_audit, treatments = verify_frozen_token_audit(adapter, config, episodes)
    value_ids = _registered_value_ids(adapter, config)
    hidden_size = int(adapter.model.config.hidden_size)
    estimated_bytes = estimate_capture_bytes(
        examples=len(episodes),
        candidates=len(candidate_sites),
        hidden_size=hidden_size,
        selected_logits=len(value_ids),
    )
    budget_bytes = int(float(config["activation_budget_mb"]) * 1024**2)
    if estimated_bytes > budget_bytes:
        raise RuntimeError(
            f"BLOCKED_RESOURCE: estimated binding capture {estimated_bytes} exceeds {budget_bytes}"
        )

    config_digest = _config_digest(config)
    runtime = _runtime_fingerprint(torch, model_metadata)
    capture_identity = {
        "config_digest": config_digest,
        "git_commit": provenance.git_commit,
        "token_audit_sha256": token_audit["episode_sha256"],
        "token_pool_sha256": token_audit["token_pool_sha256"],
        "model_revision": resolved_revision,
        "runtime": runtime,
    }
    capture_digest = hashlib.sha256(
        json.dumps(capture_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_dir = Path(str(config["output_dir"]))
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    captures: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []
    resumed = 0
    for episode_index, episode in enumerate(episodes):
        progress_path = progress_dir / f"{episode.episode_id}.npz"
        if progress_path.exists():
            captures.append(
                load_episode_progress(
                    progress_path,
                    capture_digest,
                    episode,
                    expected_treatment_site=treatment_site,
                    expected_seed=seed,
                    expected_treatment=treatments[episode.episode_id],
                )
            )
            resumed += 1
            continue
        treatment = treatments[episode.episode_id]
        arrays, record = capture_binding_episode(
            adapter,
            episode,
            treatment,
            candidate_sites=candidate_sites,
            treatment_site=treatment_site,
            final_site=final_site,
            value_ids=value_ids,
            seed=seed,
        )
        save_episode_progress(progress_path, capture_digest, episode, arrays, record)
        captures.append((arrays, record))
        if episode_index == int(config["splits"]["calibration"]["count"]) - 1:
            calibration = aggregate_capture_metrics(captures, config)
            if calibration["maximum_treatment_logit_replay_error"] > float(
                config["gates"]["treatment_logit_replay_atol"]
            ):
                raise RuntimeError("calibration treatment replay failed; protected capture stopped")

    assert_capture_contract(
        captures,
        config,
        expected_treatments=treatments,
        candidates=len(candidate_sites),
        hidden_size=hidden_size,
        selected_logits=len(value_ids),
    )
    arrays = stack_episode_captures([capture[0] for capture in captures])
    records = [capture[1] for capture in captures]
    expected_content_digest = capture_content_digest(arrays, records)
    dataset_id = str(config["capture_dataset_id"])
    storage = write_hdf5_shards(
        output_dir,
        arrays,
        records,
        dataset_id=dataset_id,
        config_digest=capture_digest,
        max_shard_mb=float(config["max_shard_mb"]),
        budget_mb=float(config["activation_budget_mb"]),
        resume=True,
    )
    stored_arrays, stored_records = read_hdf5_shards(output_dir)
    stored_content_digest = capture_content_digest(stored_arrays, stored_records)
    if stored_content_digest != expected_content_digest:
        raise RuntimeError("stored HDF5 content does not match captured progress units")
    storage = {
        **storage,
        "content_sha256": expected_content_digest,
        "readback_verified": True,
    }
    capture_metrics = aggregate_capture_metrics(captures, config)
    gates = capture_eligibility_gates(capture_metrics, config)
    if not gates["capture_integrity"]:
        status = "REJECTED_CAPTURE_INTEGRITY"
    elif not gates["treatment_replay"]:
        status = "REJECTED_TREATMENT_REPLAY"
    elif not gates["all_protected_splits_competent"]:
        status = "INELIGIBLE_TASK"
    else:
        status = "CAUSAL_DATASET_ELIGIBLE"
    metrics: dict[str, Any] = {
        "experiment_id": dataset_id,
        "parent_experiment_id": str(config["id"]),
        "status": status,
        "evidence_level": "Availability",
        "model": str(config["model"]),
        "model_revision": resolved_revision,
        "precision": str(config["dtype"]),
        "episodes": len(episodes),
        "candidate_sites": list(candidate_sites),
        "treatment_site": treatment_site,
        "final_site": final_site,
        "value_token_ids": value_ids,
        "estimated_bytes_before_capture": estimated_bytes,
        "config_digest": config_digest,
        "capture_digest": capture_digest,
        "capture_identity": capture_identity,
        "token_audit": {
            "experiment_id": token_audit["experiment_id"],
            "episode_sha256": token_audit["episode_sha256"],
            "token_pool_sha256": token_audit["token_pool_sha256"],
        },
        "resumed_episodes": resumed,
        "capture": capture_metrics,
        "gates": gates,
        "storage": storage,
        "hardware": hardware.as_dict(),
        "runtime_seconds": float(time.perf_counter() - started),
        "scientific_boundary": (
            "This artifact contains directly executed clean and upstream-treated Qwen trajectories. "
            "It does not rank mediators, decide H-LLM-15/16, validate an Intervention-JEPA, "
            "reconstruct a circuit, or establish a workspace."
        ),
    }
    metrics_path = _capture_metrics_path(config)
    manifest_path = Path(str(config["output_manifest"]))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        **storage,
        "shards": [dict(shard) for shard in storage["shards"]],
        "model": str(config["model"]),
        "model_revision": resolved_revision,
        "precision": str(config["dtype"]),
        "candidate_sites": list(candidate_sites),
        "treatment_site": treatment_site,
        "final_site": final_site,
        "value_token_ids": value_ids,
        "capture_digest": capture_digest,
        "capture_identity": capture_identity,
        "content_sha256": expected_content_digest,
        "local_data_root": output_dir.as_posix(),
    }
    for shard in manifest["shards"]:
        shard["path"] = f"{manifest['local_data_root']}/{shard['path']}"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        metrics_path.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": metrics_path.as_posix(), "status": status, "gates": gates},
    )
    return metrics


def capture_binding_episode(
    adapter: QwenHFAdapter,
    episode: BindingEpisode,
    treatment: TokenizedTreatment,
    *,
    candidate_sites: Sequence[str],
    treatment_site: str,
    final_site: str,
    value_ids: Sequence[int],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Capture one clean/donor/treated episode through direct execution."""

    recipient_batch = adapter.tokenize([episode.recipient_prompt()])
    donor_batch = adapter.tokenize([episode.donor_prompt()])
    if tuple(int(value) for value in recipient_batch.input_ids[0].tolist()) != treatment.recipient_ids:
        raise RuntimeError("recipient token IDs changed after the frozen token audit")
    if tuple(int(value) for value in donor_batch.input_ids[0].tolist()) != treatment.donor_ids:
        raise RuntimeError("donor token IDs changed after the frozen token audit")
    clean_sites = [*candidate_sites, treatment_site, final_site, "logits"]
    clean = adapter.forward_with_cache(recipient_batch, clean_sites)
    donor = adapter.forward_with_cache(donor_batch, [treatment_site, "logits"])
    donor_id = f"{episode.episode_id}:treatment"
    adapter.register_donor(donor_id, treatment_site, donor.activations[treatment_site])
    spec = InterventionSpec(
        site=treatment_site,
        operation="patch",
        positions=treatment.changed_positions,
        donor_example_id=donor_id,
        seed=seed,
    )
    try:
        treated = adapter.forward_with_intervention(recipient_batch, spec, clean_sites)
    finally:
        adapter.unregister_donor(donor_id, treatment_site)
    selected = np.asarray(value_ids, dtype=np.int64)
    clean_logits = np.asarray(clean.logits[0, -1], dtype=np.float32)
    donor_logits = np.asarray(donor.logits[0, -1], dtype=np.float32)
    treated_logits = np.asarray(treated.logits[0, -1], dtype=np.float32)
    clean_score = float(
        clean_logits[treatment.donor_answer_id] - clean_logits[treatment.recipient_answer_id]
    )
    treated_score = float(
        treated_logits[treatment.donor_answer_id] - treated_logits[treatment.recipient_answer_id]
    )
    arrays = {
        "clean_candidate": np.stack(
            [np.asarray(clean.activations[site][0, -1], dtype=np.float32) for site in candidate_sites]
        ),
        "treated_candidate": np.stack(
            [np.asarray(treated.activations[site][0, -1], dtype=np.float32) for site in candidate_sites]
        ),
        "clean_final": np.asarray(clean.activations[final_site][0, -1], dtype=np.float32),
        "treated_final": np.asarray(treated.activations[final_site][0, -1], dtype=np.float32),
        "clean_value_logits": clean_logits[selected],
        "donor_value_logits": donor_logits[selected],
        "treated_value_logits": treated_logits[selected],
        "recipient_answer_id": np.asarray(treatment.recipient_answer_id, dtype=np.int64),
        "donor_answer_id": np.asarray(treatment.donor_answer_id, dtype=np.int64),
        "clean_top_token": np.asarray(clean_logits.argmax(), dtype=np.int64),
        "donor_top_token": np.asarray(donor_logits.argmax(), dtype=np.int64),
        "treated_top_token": np.asarray(treated_logits.argmax(), dtype=np.int64),
        "clean_score": np.asarray(clean_score, dtype=np.float32),
        "treated_score": np.asarray(treated_score, dtype=np.float32),
        "treatment_effect": np.asarray(treated_score - clean_score, dtype=np.float32),
        "treatment_source_replay_error": np.asarray(
            np.max(
                np.abs(
                    np.asarray(treated.activations[treatment_site], dtype=np.float32)
                    - np.asarray(donor.activations[treatment_site], dtype=np.float32)
                )
            ),
            dtype=np.float32,
        ),
        "treatment_logit_replay_error": np.asarray(
            np.max(
                np.abs(
                    np.asarray(treated.logits, dtype=np.float32)
                    - np.asarray(donor.logits, dtype=np.float32)
                )
            ),
            dtype=np.float32,
        ),
    }
    record = {
        "example_id": episode.episode_id,
        "episode": episode.to_dict(),
        "changed_positions": list(treatment.changed_positions),
        "recipient_answer_id": treatment.recipient_answer_id,
        "donor_answer_id": treatment.donor_answer_id,
        "intervention": spec.to_dict(),
    }
    return arrays, record


def estimate_capture_bytes(
    *, examples: int, candidates: int, hidden_size: int, selected_logits: int
) -> int:
    """Conservative uncompressed estimate for the selected capture arrays."""

    causal_float_values = examples * (2 * candidates * hidden_size + 2 * hidden_size)
    float_values = examples * (3 * selected_logits + 5)
    integer_values = examples * 5
    raw_arrays = causal_float_values * 4 + float_values * 4 + integer_values * 8
    # Progress units coexist with final shards until scientific aggregation is
    # committed.  Budget both copies plus metadata/checksum overhead.
    return int(raw_arrays * 2.20)


def capture_content_digest(
    arrays: Mapping[str, np.ndarray], records: Sequence[Mapping[str, Any]]
) -> str:
    """Hash array contents, dtype/shape, and ordered metadata records."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        descriptor = {
            "name": name,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
        digest.update(json.dumps(descriptor, sort_keys=True).encode("utf-8"))
        digest.update(value.tobytes(order="C"))
    digest.update(
        json.dumps(list(records), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def assert_capture_contract(
    captures: Sequence[tuple[Mapping[str, np.ndarray], Mapping[str, Any]]],
    config: Mapping[str, Any],
    *,
    expected_treatments: Mapping[str, TokenizedTreatment],
    candidates: int,
    hidden_size: int,
    selected_logits: int,
) -> None:
    """Fail closed on missing, duplicate, non-finite, lossy, or malformed rows."""

    expected_examples = sum(int(spec["count"]) for spec in config["splits"].values())
    if len(captures) != expected_examples:
        raise RuntimeError(f"expected {expected_examples} captures, got {len(captures)}")
    expected_episodes = binding_episodes_from_config(config)
    identifiers = [str(record.get("example_id")) for _arrays, record in captures]
    if identifiers != [episode.episode_id for episode in expected_episodes]:
        raise RuntimeError("capture IDs/order differ from the canonical episode roster")
    expected_shapes = {
        "clean_candidate": (candidates, hidden_size),
        "treated_candidate": (candidates, hidden_size),
        "clean_final": (hidden_size,),
        "treated_final": (hidden_size,),
        "clean_value_logits": (selected_logits,),
        "donor_value_logits": (selected_logits,),
        "treated_value_logits": (selected_logits,),
        "recipient_answer_id": (),
        "donor_answer_id": (),
        "clean_top_token": (),
        "donor_top_token": (),
        "treated_top_token": (),
        "clean_score": (),
        "treated_score": (),
        "treatment_effect": (),
        "treatment_source_replay_error": (),
        "treatment_logit_replay_error": (),
    }
    integer_names = {
        "recipient_answer_id",
        "donor_answer_id",
        "clean_top_token",
        "donor_top_token",
        "treated_top_token",
    }
    float_names = set(expected_shapes).difference(integer_names)
    for (arrays, record), episode in zip(captures, expected_episodes, strict=True):
        expected_treatment = expected_treatments.get(episode.episode_id)
        if expected_treatment is None:
            raise RuntimeError(
                f"missing frozen tokenized treatment for {episode.episode_id}"
            )
        if set(arrays) != set(expected_shapes):
            raise RuntimeError(f"{episode.episode_id} capture array schema differs")
        _validate_progress_record(
            record,
            episode,
            expected_treatment_site=str(config["treatment"]["site"]),
            expected_seed=int(config["seed"]),
            expected_treatment=expected_treatment,
        )
        split = record.get("episode", {}).get("split")
        if split not in config["splits"]:
            raise RuntimeError(f"unknown capture split {split!r}")
        for name, shape in expected_shapes.items():
            value = np.asarray(arrays.get(name))
            if value.shape != shape:
                raise RuntimeError(
                    f"{record['example_id']} {name} shape {value.shape} != {shape}"
                )
        for name in float_names:
            if np.asarray(arrays[name]).dtype != np.float32:
                raise RuntimeError(f"{record['example_id']} {name} must be float32")
        for name in integer_names:
            if np.asarray(arrays[name]).dtype != np.int64:
                raise RuntimeError(f"{record['example_id']} {name} must be int64")
        _validate_capture_answer_ids(arrays, record, expected_treatment, episode)
        _assert_finite_arrays(arrays, example_id=str(record["example_id"]))


def save_episode_progress(
    path: Path,
    config_digest: str,
    episode: BindingEpisode,
    arrays: Mapping[str, np.ndarray],
    record: Mapping[str, Any],
) -> None:
    """Atomically save one resumable capture unit."""

    _validate_progress_record(record, episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config_digest": config_digest,
        "episode": episode.to_dict(),
        "record": record,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
    temporary.replace(path)


def load_episode_progress(
    path: Path,
    config_digest: str,
    episode: BindingEpisode,
    *,
    expected_treatment_site: str | None = None,
    expected_seed: int | None = None,
    expected_treatment: TokenizedTreatment | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a progress unit only when config and episode identity match."""

    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        if metadata["config_digest"] != config_digest:
            raise RuntimeError(f"stale capture config in {path}")
        if json.dumps(metadata["episode"], sort_keys=True) != json.dumps(
            episode.to_dict(), sort_keys=True
        ):
            raise RuntimeError(f"stale capture episode in {path}")
        record = dict(metadata["record"])
        _validate_progress_record(
            record,
            episode,
            expected_treatment_site=expected_treatment_site,
            expected_seed=expected_seed,
            expected_treatment=expected_treatment,
        )
        arrays = {name: payload[name].copy() for name in payload.files if name != "metadata"}
        if expected_treatment is not None:
            _validate_capture_answer_ids(arrays, record, expected_treatment, episode)
    return arrays, record


def _validate_progress_record(
    record: Mapping[str, Any],
    episode: BindingEpisode,
    *,
    expected_treatment_site: str | None = None,
    expected_seed: int | None = None,
    expected_treatment: TokenizedTreatment | None = None,
) -> None:
    if record.get("example_id") != episode.episode_id:
        raise RuntimeError("capture record example ID differs from its episode")
    if json.dumps(record.get("episode"), sort_keys=True) != json.dumps(
        episode.to_dict(), sort_keys=True
    ):
        raise RuntimeError("capture record episode differs from the canonical episode")
    if (
        expected_treatment_site is None
        and expected_seed is None
        and expected_treatment is None
    ):
        return
    if (
        expected_treatment_site is None
        or expected_seed is None
        or expected_treatment is None
    ):
        raise ValueError("site, seed, and tokenized treatment must be validated together")
    intervention_payload = record.get("intervention")
    if not isinstance(intervention_payload, Mapping):
        raise RuntimeError("capture record is missing its frozen intervention")
    intervention = InterventionSpec.from_dict(intervention_payload)
    changed_positions = tuple(int(value) for value in record.get("changed_positions", ()))
    expected_donor = f"{episode.episode_id}:treatment"
    if (
        intervention.site != expected_treatment_site
        or intervention.operation != "patch"
        or changed_positions != expected_treatment.changed_positions
        or intervention.positions != expected_treatment.changed_positions
        or intervention.feature_ids is not None
        or intervention.magnitude != 1.0
        or intervention.donor_example_id != expected_donor
        or intervention.seed != expected_seed
    ):
        raise RuntimeError("capture record intervention differs from the frozen treatment")
    if (
        int(record.get("recipient_answer_id", -1))
        != expected_treatment.recipient_answer_id
        or int(record.get("donor_answer_id", -1)) != expected_treatment.donor_answer_id
    ):
        raise RuntimeError("capture record answer IDs differ from the frozen treatment")


def _validate_capture_answer_ids(
    arrays: Mapping[str, np.ndarray],
    record: Mapping[str, Any],
    treatment: TokenizedTreatment | None,
    episode: BindingEpisode,
) -> None:
    if treatment is None:
        raise RuntimeError(f"missing frozen tokenized treatment for {episode.episode_id}")
    expected = {
        "recipient_answer_id": treatment.recipient_answer_id,
        "donor_answer_id": treatment.donor_answer_id,
    }
    for name, value in expected.items():
        array_value = np.asarray(arrays[name])
        if array_value.shape != ():
            raise RuntimeError(f"{episode.episode_id} {name} must be scalar")
        if int(record.get(name, -1)) != value or int(array_value.item()) != value:
            raise RuntimeError(f"{episode.episode_id} {name} differs from frozen treatment")


def stack_episode_captures(captures: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack identically shaped progress units into aligned dataset arrays."""

    if not captures:
        raise ValueError("at least one capture is required")
    names = set(captures[0])
    if any(set(capture) != names for capture in captures):
        raise ValueError("episode capture schemas differ")
    return {name: np.stack([capture[name] for capture in captures]) for name in sorted(names)}


def aggregate_capture_metrics(
    captures: Sequence[tuple[Mapping[str, np.ndarray], Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate competence and replay by independent episode split."""

    if not captures:
        raise ValueError("at least one capture is required")
    for arrays, record in captures:
        _assert_finite_arrays(arrays, example_id=str(record.get("example_id", "UNKNOWN")))
    identifiers = [str(record.get("example_id")) for _arrays, record in captures]
    ids_unique = len(set(identifiers)) == len(identifiers) and "None" not in identifiers
    by_split: dict[str, dict[str, Any]] = {}
    group_accuracy: dict[str, dict[str, Any]] = {}
    for split in config["splits"]:
        rows = [(arrays, record) for arrays, record in captures if record["episode"]["split"] == split]
        if not rows:
            continue
        clean_correct = [
            int(arrays["clean_top_token"]) == int(arrays["recipient_answer_id"])
            for arrays, _record in rows
        ]
        donor_correct = [
            int(arrays["donor_top_token"]) == int(arrays["donor_answer_id"])
            for arrays, _record in rows
        ]
        transfer = [
            int(arrays["treated_top_token"]) == int(arrays["donor_answer_id"])
            for arrays, _record in rows
        ]
        by_split[split] = {
            "episodes": len(rows),
            "clean_accuracy": float(np.mean(clean_correct)),
            "donor_accuracy": float(np.mean(donor_correct)),
            "treatment_donor_transfer": float(np.mean(transfer)),
            "mean_absolute_treatment_effect": float(
                np.mean([abs(float(arrays["treatment_effect"])) for arrays, _record in rows])
            ),
            "mean_treatment_effect": float(
                np.mean([float(arrays["treatment_effect"]) for arrays, _record in rows])
            ),
        }
        group_accuracy[split] = _group_accuracy(rows)
    expected_counts = {
        split: int(spec["count"]) for split, spec in config["splits"].items()
    }
    observed_counts = {
        split: int(by_split.get(split, {}).get("episodes", 0)) for split in expected_counts
    }
    source_replay = np.asarray(
        [float(arrays["treatment_source_replay_error"]) for arrays, _record in captures],
        dtype=np.float64,
    )
    logit_replay = np.asarray(
        [float(arrays["treatment_logit_replay_error"]) for arrays, _record in captures],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(source_replay)) or not np.all(np.isfinite(logit_replay)):
        raise RuntimeError("non-finite treatment replay error")
    return {
        "by_split": by_split,
        "group_accuracy": group_accuracy,
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "exact_counts": observed_counts == expected_counts,
        "unique_example_ids": ids_unique,
        "all_arrays_finite": True,
        "maximum_treatment_source_replay_error": float(np.max(source_replay)),
        "maximum_treatment_logit_replay_error": float(np.max(logit_replay)),
    }


def capture_eligibility_gates(
    metrics: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, bool]:
    """Apply preregistered task/replay gates without calibration leakage."""

    threshold = config["gates"]
    protected = ("train", "validation", "test", "paraphrase")
    by_split = metrics["by_split"]
    competence = {
        split: bool(
            by_split.get(split, {}).get("clean_accuracy", -np.inf)
            >= float(threshold["clean_accuracy_min_each_split"])
            and by_split.get(split, {}).get("donor_accuracy", -np.inf)
            >= float(threshold.get("donor_accuracy_min_each_split", 0.0))
            and by_split.get(split, {}).get("treatment_donor_transfer", -np.inf)
            >= float(threshold["full_treatment_donor_transfer_min_each_split"])
            and by_split.get(split, {}).get("mean_absolute_treatment_effect", -np.inf)
            >= float(threshold.get("treatment_effect_absolute_min", 0.0))
        )
        for split in protected
    }
    group_competence = {
        split: bool(
            metrics["group_accuracy"].get(split, {}).get("minimum_clean_accuracy", -np.inf)
            >= float(threshold.get("minimum_key_or_value_clean_accuracy_min", 0.0))
            and metrics["group_accuracy"].get(split, {}).get(
                "minimum_donor_accuracy", -np.inf
            )
            >= float(threshold.get("minimum_key_or_value_donor_accuracy_min", 0.0))
            and metrics["group_accuracy"].get(split, {}).get("minimum_transfer", -np.inf)
            >= float(threshold.get("minimum_key_or_value_transfer_min", 0.0))
        )
        for split in protected
    }
    integrity = bool(
        metrics.get("exact_counts")
        and metrics.get("unique_example_ids")
        and metrics.get("all_arrays_finite")
    )
    return {
        "capture_integrity": integrity,
        "treatment_replay": bool(
            integrity
            and metrics["maximum_treatment_logit_replay_error"]
            <= float(threshold["treatment_logit_replay_atol"])
            and metrics["maximum_treatment_source_replay_error"]
            <= float(threshold["treatment_logit_replay_atol"])
        ),
        **{f"{split}_competent": value for split, value in competence.items()},
        **{f"{split}_groups_competent": value for split, value in group_competence.items()},
        "all_protected_splits_competent": bool(
            integrity and all(competence.values()) and all(group_competence.values())
        ),
    }


def _assert_finite_arrays(
    arrays: Mapping[str, np.ndarray], *, example_id: str
) -> None:
    for name, value in arrays.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise RuntimeError(f"non-finite capture array {name!r} in {example_id}")


def _group_accuracy(
    rows: Sequence[tuple[Mapping[str, np.ndarray], Mapping[str, Any]]],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[tuple[bool, bool, bool]]]] = {
        "query_key": {},
        "recipient_value": {},
        "donor_value": {},
    }
    for arrays, record in rows:
        episode = record["episode"]
        query_index = int(episode["query_index"])
        labels = {
            "query_key": str(episode["keys"][query_index]),
            "recipient_value": str(episode["recipient_values"][query_index]),
            "donor_value": str(episode["donor_values"][query_index]),
        }
        outcomes = (
            int(arrays["clean_top_token"]) == int(arrays["recipient_answer_id"]),
            int(arrays["donor_top_token"]) == int(arrays["donor_answer_id"]),
            int(arrays["treated_top_token"]) == int(arrays["donor_answer_id"]),
        )
        for family, label in labels.items():
            grouped[family].setdefault(label, []).append(outcomes)
    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for family, groups in grouped.items():
        summaries[family] = {
            label: {
                "episodes": len(outcomes),
                "clean_accuracy": float(np.mean([item[0] for item in outcomes])),
                "donor_accuracy": float(np.mean([item[1] for item in outcomes])),
                "treatment_donor_transfer": float(
                    np.mean([item[2] for item in outcomes])
                ),
            }
            for label, outcomes in sorted(groups.items())
        }
    clean_groups = [
        details["clean_accuracy"]
        for family in ("query_key", "recipient_value")
        for details in summaries[family].values()
    ]
    donor_groups = [
        details["donor_accuracy"]
        for family in ("query_key", "donor_value")
        for details in summaries[family].values()
    ]
    transfer_groups = [
        details["treatment_donor_transfer"]
        for family in ("query_key", "donor_value")
        for details in summaries[family].values()
    ]
    return {
        "families": summaries,
        "macro_clean_accuracy": float(np.mean(clean_groups)),
        "macro_donor_accuracy": float(np.mean(donor_groups)),
        "macro_transfer": float(np.mean(transfer_groups)),
        "minimum_clean_accuracy": float(np.min(clean_groups)),
        "minimum_donor_accuracy": float(np.min(donor_groups)),
        "minimum_transfer": float(np.min(transfer_groups)),
    }


def verify_frozen_token_audit(
    adapter: QwenHFAdapter,
    config: Mapping[str, Any],
    episodes: Sequence[BindingEpisode],
) -> tuple[dict[str, Any], dict[str, TokenizedTreatment]]:
    """Recompute and bind the protected run to a successful token audit."""

    audit_path = Path(str(config["tokenization_audit_metrics"]))
    if not audit_path.exists():
        raise RuntimeError(f"frozen token audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": str(config["tokenization_audit_id"]),
        "parent_experiment_id": str(config["id"]),
        "model": str(config["model"]),
        "revision": str(config["revision"]),
        "resolved_revision": str(config["revision"]),
        "episode_count": len(episodes),
    }
    for field, expected_value in expected.items():
        if audit.get(field) != expected_value:
            raise RuntimeError(
                f"token audit {field} mismatch: {audit.get(field)!r} != {expected_value!r}"
            )
    if audit.get("status") != "SMOKE_VALIDATED" or not all(audit.get("gates", {}).values()):
        raise RuntimeError("token audit did not pass every frozen gate")
    _pool_payload, pool_single, pool_disjoint, pool_digest = audit_token_pools(
        config["token_pools"],
        encode_item=lambda item: adapter.tokenizer.encode(
            f" {item}", add_special_tokens=False
        ),
    )
    if not pool_single or not pool_disjoint:
        raise RuntimeError("live tokenizer violates the frozen token-pool contract")
    if pool_digest != audit.get("token_pool_sha256"):
        raise RuntimeError("live tokenizer does not reproduce the frozen token-pool digest")
    treatments: dict[str, TokenizedTreatment] = {}
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        treatment = _audit_episode(adapter, episode)
        treatments[episode.episode_id] = treatment
        rows.append(tokenized_treatment_payload(episode, treatment))
    recomputed = tokenization_digest(rows)
    if recomputed != audit.get("episode_sha256"):
        raise RuntimeError("live tokenizer does not reproduce the frozen episode digest")
    return audit, treatments


def _runtime_fingerprint(torch: Any, model_metadata: Mapping[str, Any]) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "numpy": str(np.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0))
        if cuda_available
        else None,
        "model_type": model_metadata.get("model_type"),
        "attention_implementation": model_metadata.get("attention_implementation"),
        "dtype": model_metadata.get("dtype"),
    }


def _audit_episode(adapter: QwenHFAdapter, episode: BindingEpisode) -> TokenizedTreatment:
    return audit_tokenized_treatment(
        episode,
        encode_prompt=lambda prompt: adapter.tokenizer.encode(prompt, add_special_tokens=True),
        encode_answer=lambda answer: adapter.tokenizer.encode(
            f" {answer}", add_special_tokens=False
        ),
    )


def _registered_value_ids(adapter: QwenHFAdapter, config: Mapping[str, Any]) -> list[int]:
    values = [
        value
        for split in ("calibration", "train", "validation", "test")
        for value in config["token_pools"]["values"][split]
    ]
    ids: list[int] = []
    for value in values:
        encoded = adapter.tokenizer.encode(f" {value}", add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"registered value {value!r} is no longer one token")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise RuntimeError("registered values do not map to unique token IDs")
    return ids


def _config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _capture_metrics_path(config: Mapping[str, Any]) -> Path:
    final = Path(str(config["output_metrics"]))
    return final.with_name(final.name.replace("mediation", "capture"))
