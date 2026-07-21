"""Activation storage estimates and lightweight manifests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal_workspace_jepa.common.resources import estimate_activation_bytes


@dataclass(frozen=True)
class ActivationStorageEstimate:
    examples: int
    layers: int
    positions: int
    hidden_size: int
    bytes_per_value: int
    estimated_bytes: int


def estimate_storage(
    examples: int,
    layers: int,
    positions: int,
    hidden_size: int,
    bytes_per_value: int = 2,
) -> ActivationStorageEstimate:
    return ActivationStorageEstimate(
        examples=examples,
        layers=layers,
        positions=positions,
        hidden_size=hidden_size,
        bytes_per_value=bytes_per_value,
        estimated_bytes=estimate_activation_bytes(
            examples=examples,
            layers=layers,
            positions=positions,
            hidden_size=hidden_size,
            bytes_per_value=bytes_per_value,
        ),
    )


def write_hdf5_shards(
    output_dir: str | Path,
    arrays: Mapping[str, np.ndarray],
    records: list[Mapping[str, Any]],
    *,
    dataset_id: str,
    config_digest: str,
    max_shard_mb: float = 256.0,
    budget_mb: float = 1024.0,
    resume: bool = True,
) -> dict[str, Any]:
    """Write aligned arrays and JSON metadata to checksummed HDF5 shards.

    Existing complete shards are reused only when the dataset/config identity and
    every checksum match. Writes use a temporary sibling followed by an atomic
    rename, so interrupted runs never masquerade as complete shards.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional GPU dependency
        raise RuntimeError("HDF5 activation storage requires h5py") from exc
    if not arrays:
        raise ValueError("at least one array is required")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    lengths = {value.shape[0] for value in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("all activation arrays must share the first dimension")
    examples = lengths.pop()
    if len(records) != examples:
        raise ValueError("metadata record count must equal array example count")
    per_example = sum(max(1, value[0].nbytes) for value in normalized.values())
    metadata_bytes = sum(len(json.dumps(record, sort_keys=True)) for record in records)
    estimated_bytes = sum(value.nbytes for value in normalized.values()) + metadata_bytes
    budget_bytes = int(budget_mb * 1024**2)
    if estimated_bytes > budget_bytes:
        raise RuntimeError(
            f"BLOCKED_RESOURCE: activation dataset estimate {estimated_bytes} exceeds "
            f"budget {budget_bytes}"
        )
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != dataset_id or manifest.get("config_digest") != config_digest:
            raise RuntimeError("existing activation manifest does not match dataset/config identity")
        if all(_sha256(output / shard["path"]) == shard["sha256"] for shard in manifest["shards"]):
            return manifest
        raise RuntimeError("existing activation shard checksum mismatch")
    output.mkdir(parents=True, exist_ok=True)
    shard_bytes = max(1, int(max_shard_mb * 1024**2))
    rows_per_shard = max(1, shard_bytes // max(per_example, 1))
    shard_count = math.ceil(examples / rows_per_shard)
    shards: list[dict[str, Any]] = []
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for shard_index, start in enumerate(range(0, examples, rows_per_shard)):
        stop = min(start + rows_per_shard, examples)
        name = f"shard-{shard_index:05d}-of-{shard_count:05d}.h5"
        final_path = output / name
        temporary_path = output / f".{name}.partial"
        with h5py.File(temporary_path, "w") as handle:
            handle.attrs["dataset_id"] = dataset_id
            handle.attrs["config_digest"] = config_digest
            handle.attrs["start"] = start
            handle.attrs["stop"] = stop
            group = handle.create_group("arrays")
            for array_name, value in normalized.items():
                group.create_dataset(array_name, data=value[start:stop], compression="gzip")
            serialized = [json.dumps(record, sort_keys=True) for record in records[start:stop]]
            handle.create_dataset("records_json", data=np.asarray(serialized, dtype=object), dtype=string_dtype)
        temporary_path.replace(final_path)
        shards.append(
            {
                "path": name,
                "rows": stop - start,
                "bytes": final_path.stat().st_size,
                "sha256": _sha256(final_path),
            }
        )
    manifest = {
        "dataset_id": dataset_id,
        "config_digest": config_digest,
        "format": "hdf5",
        "examples": examples,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in normalized.items()
        },
        "estimated_uncompressed_bytes": estimated_bytes,
        "budget_bytes": budget_bytes,
        "max_shard_bytes": shard_bytes,
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_hdf5_shards(output_dir: str | Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Read and checksum-validate a dataset written by :func:`write_hdf5_shards`."""

    import h5py

    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in manifest["arrays"]}
    records: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        path = output / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise RuntimeError(f"activation shard checksum mismatch: {path}")
        with h5py.File(path, "r") as handle:
            for name in chunks:
                chunks[name].append(handle["arrays"][name][...])
            records.extend(json.loads(value) for value in handle["records_json"].asstr()[...])
    return {name: np.concatenate(value, axis=0) for name, value in chunks.items()}, records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
