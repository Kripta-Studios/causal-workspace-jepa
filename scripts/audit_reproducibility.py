#!/usr/bin/env python
"""Audit repository scaffolding, experiment provenance, and local checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_PATHS = [
    "README.md",
    "SUMMARY.md",
    "configs/resource/cpu_vps.yaml",
    "docs/EXPERIMENT_REGISTRY.md",
    "docs/RESULTS.md",
    "data/manifests/datasets.yaml",
    "papers/references.bib",
]


def main() -> int:
    missing = [path for path in REQUIRED_PATHS if not Path(path).exists()]
    errors: list[str] = []
    metric_count = _audit_metrics(errors)
    verified_checksums, skipped_checksums = _audit_manifests(errors)
    if missing:
        errors.extend(f"missing required path: {path}" for path in missing)
    payload = {
        "status": "SMOKE_VALIDATED" if not errors else "BLOCKED",
        "missing": missing,
        "errors": errors,
        "metric_artifacts": metric_count,
        "checksums_verified": verified_checksums,
        "checksums_skipped_missing_local_data": skipped_checksums,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 2


def _audit_metrics(errors: list[str]) -> int:
    count = 0
    for metrics_path in sorted(Path("artifacts/metrics").glob("*.json")):
        if metrics_path.name.endswith(".provenance.json"):
            continue
        count += 1
        metrics = _load_json(metrics_path, errors)
        if metrics is None:
            continue
        for field in ("experiment_id", "status", "evidence_level"):
            if field not in metrics:
                errors.append(f"{metrics_path}: missing {field}")
        provenance_path = metrics_path.with_suffix(".provenance.json")
        provenance = _load_json(provenance_path, errors)
        if provenance is None:
            continue
        if provenance.get("git_dirty") is not False:
            errors.append(f"{provenance_path}: git_dirty must be false")
        commit = provenance.get("git_commit")
        if not isinstance(commit, str) or len(commit) < 7:
            errors.append(f"{provenance_path}: invalid git_commit")
        if provenance.get("metrics") != str(metrics_path):
            errors.append(f"{provenance_path}: metrics path does not match {metrics_path}")
    return count


def _audit_manifests(errors: list[str]) -> tuple[int, int]:
    verified = 0
    skipped = 0
    for manifest_path in sorted(Path("data/manifests").glob("*.json")):
        manifest = _load_json(manifest_path, errors)
        if manifest is None:
            continue
        for record in _checksum_records(manifest):
            data_path = Path(str(record["path"]))
            if not data_path.exists():
                skipped += 1
                continue
            expected = str(record["sha256"])
            actual = _sha256(data_path)
            if actual != expected:
                errors.append(
                    f"{manifest_path}: checksum mismatch for {data_path}: {actual} != {expected}"
                )
            else:
                verified += 1
    return verified, skipped


def _checksum_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            records.append(value)
        for nested in value.values():
            records.extend(_checksum_records(nested))
    elif isinstance(value, list):
        for nested in value:
            records.extend(_checksum_records(nested))
    return records


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"missing JSON file: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path}: top-level JSON must be an object")
        return None
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
