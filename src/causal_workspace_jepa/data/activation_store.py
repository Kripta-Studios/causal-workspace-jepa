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
    for name, value in normalized.items():
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"activation array must be numeric: {name}")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"activation array contains nonfinite values: {name}")
    lengths = {value.shape[0] for value in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("all activation arrays must share the first dimension")
    examples = lengths.pop()
    if examples <= 0:
        raise ValueError("activation datasets must contain at least one example")
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
        _validate_manifest_structure(manifest, output)
        if all(_sha256(output / shard["path"]) == shard["sha256"] for shard in manifest["shards"]):
            _validate_existing_content(manifest, output, normalized, records)
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
    _validate_manifest_structure(manifest, output)
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
    arrays = {name: np.concatenate(value, axis=0) for name, value in chunks.items()}
    if len(records) != int(manifest["examples"]) or any(
        value.shape[0] != int(manifest["examples"]) for value in arrays.values()
    ):
        raise RuntimeError("activation shards do not reconstruct the manifest example count")
    return arrays, records


def _validate_manifest_structure(manifest: Mapping[str, Any], output: Path) -> None:
    import h5py

    shards = manifest.get("shards")
    arrays = manifest.get("arrays")
    examples = manifest.get("examples")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("activation manifest must contain at least one shard")
    if not isinstance(arrays, Mapping) or not arrays:
        raise RuntimeError("activation manifest must describe at least one array")
    if not isinstance(examples, int) or examples <= 0:
        raise RuntimeError("activation manifest example count must be positive")
    observed_rows = 0
    observed_paths: set[str] = set()
    expected_start = 0
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise RuntimeError("activation manifest contains a malformed shard record")
        relative = Path(str(shard.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or relative.name != str(relative):
            raise RuntimeError("activation shard path must be a simple relative filename")
        path = output / relative
        if not path.is_file():
            raise RuntimeError(f"activation shard is missing: {path}")
        if str(relative) in observed_paths:
            raise RuntimeError(f"activation manifest repeats a shard path: {relative}")
        observed_paths.add(str(relative))
        rows = shard.get("rows")
        if not isinstance(rows, int) or rows <= 0:
            raise RuntimeError("activation shard row count must be positive")
        observed_rows += rows
        if not isinstance(shard.get("sha256"), str) or len(shard["sha256"]) != 64:
            raise RuntimeError("activation shard checksum is malformed")
        with h5py.File(path, "r") as handle:
            start = int(handle.attrs.get("start", -1))
            stop = int(handle.attrs.get("stop", -1))
            if start != expected_start or stop != start + rows:
                raise RuntimeError("activation shard ranges are not contiguous and ordered")
            if handle.attrs.get("dataset_id") != manifest.get("dataset_id") or handle.attrs.get(
                "config_digest"
            ) != manifest.get("config_digest"):
                raise RuntimeError("activation shard identity differs from its manifest")
            if "arrays" not in handle or set(handle["arrays"]) != set(arrays):
                raise RuntimeError("activation shard array names differ from its manifest")
            for name, schema in arrays.items():
                if not isinstance(schema, Mapping):
                    raise RuntimeError("activation manifest contains a malformed array schema")
                dataset = handle["arrays"][name]
                expected_shape = tuple(int(value) for value in schema.get("shape", ()))
                if not expected_shape or dataset.shape != (rows, *expected_shape[1:]):
                    raise RuntimeError("activation shard array shape differs from its manifest")
                if str(dataset.dtype) != str(schema.get("dtype")):
                    raise RuntimeError("activation shard array dtype differs from its manifest")
            if "records_json" not in handle or handle["records_json"].shape != (rows,):
                raise RuntimeError("activation shard record count differs from its manifest")
        expected_start = stop
    if observed_rows != examples:
        raise RuntimeError("activation shard row counts differ from manifest examples")


def _validate_existing_content(
    manifest: Mapping[str, Any],
    output: Path,
    arrays: Mapping[str, np.ndarray],
    records: list[Mapping[str, Any]],
) -> None:
    """Prove that a resume request is byte-semantically the existing dataset."""

    import h5py

    expected_schema = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    if int(manifest["examples"]) != len(records) or manifest["arrays"] != expected_schema:
        raise RuntimeError("existing activation manifest schema differs from requested content")
    cursor = 0
    for shard in manifest["shards"]:
        rows = int(shard["rows"])
        with h5py.File(output / shard["path"], "r") as handle:
            for name, value in arrays.items():
                if not np.array_equal(handle["arrays"][name][...], value[cursor : cursor + rows]):
                    raise RuntimeError(
                        f"existing activation content differs from resume request: {name}"
                    )
            observed_records = [
                json.loads(value) for value in handle["records_json"].asstr()[...]
            ]
            expected_records = [dict(value) for value in records[cursor : cursor + rows]]
            if observed_records != expected_records:
                raise RuntimeError("existing activation records differ from resume request")
        cursor += rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
