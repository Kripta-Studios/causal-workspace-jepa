"""Frozen source-config contract for the official EB-JEPA Two Rooms training portfolio."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_CONFIG: dict[str, Any] = {
    "data.batch_size": 384,
    "data.num_workers": 16,
    "data.pin_mem": True,
    "data.persistent_workers": True,
    "training.use_amp": True,
    "training.dtype": "bfloat16",
    "model.compile": True,
    "model.dobs": 2,
    "model.henc": 32,
    "model.hpre": 32,
    "model.dstc": 32,
    "model.nsteps": 8,
    "model.encoder_architecture": "impala",
    "model.train_rollout": "last",
    "model.regularizer.cov_coeff": 8,
    "model.regularizer.std_coeff": 16,
    "model.regularizer.sim_coeff_t": 12,
    "model.regularizer.idm_coeff": 1,
    "model.regularizer.use_proj": False,
    "optim.epochs": 12,
    "optim.lr": 0.001,
    "optim.grad_clip_enc": 2.0,
    "optim.grad_clip_pred": 2.0,
    "optim.weight_decay": 1e-5,
}


def _nested_get(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        value = value[part]
    return value


def validate_source_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact frozen values, raising if the pinned upstream config drifts."""

    observed = {path: _nested_get(config, path) for path in EXPECTED_SOURCE_CONFIG}
    mismatches = {
        path: {"expected": expected, "observed": observed[path]}
        for path, expected in EXPECTED_SOURCE_CONFIG.items()
        if observed[path] != expected
    }
    if mismatches:
        raise ValueError(f"upstream training config drifted: {mismatches}")
    return observed


def required_checkpoint_names(epochs: int = 12) -> list[str]:
    return [f"e-{epoch}.pth.tar" for epoch in range(epochs)] + ["latest.pth.tar"]


def validate_completed_training_status(
    status: Mapping[str, Any],
    run_directory: str | Path,
    *,
    experiment_id: str,
    seed: int,
    source_revision: str,
    source_config: Mapping[str, Any],
    epochs: int,
    checkpoint_epoch_reader: Any,
) -> dict[str, dict[str, Any]]:
    """Recompute every checkpoint identity before accepting a completed seed.

    A JSON status file is not evidence that its checkpoint bytes still exist or
    correspond to the epoch encoded in each filename.  This validator is used
    both when a portfolio skips an already completed seed and after a new
    training run returns.
    """

    if status.get("status") != "COMPLETED":
        raise RuntimeError("training status is not COMPLETED")
    identity = {
        "experiment_id": experiment_id,
        "seed": int(seed),
        "source_revision": source_revision,
        "source_config": dict(source_config),
    }
    observed_identity = {
        "experiment_id": status.get("experiment_id"),
        "seed": int(status.get("seed", -1)),
        "source_revision": status.get("source_revision"),
        "source_config": status.get("source_config"),
    }
    if observed_identity != identity:
        raise RuntimeError("completed training status identity differs from the frozen run")
    if bool(status.get("repo_dirty_at_start")) or not bool(status.get("source_clean")):
        raise RuntimeError("completed training status did not start from clean sources")

    expected_names = required_checkpoint_names(epochs)
    manifest = status.get("checkpoint_manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != set(expected_names):
        raise RuntimeError("completed checkpoint manifest has missing or extra names")
    root = Path(run_directory)
    verified: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"completed checkpoint is missing: {path}")
        size = path.stat().st_size
        digest = _sha256(path)
        record = manifest[name]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"checkpoint manifest record is malformed: {name}")
        if int(record.get("bytes", -1)) != size or record.get("sha256") != digest:
            raise RuntimeError(f"checkpoint bytes/hash differ from completed manifest: {name}")
        expected_epoch = epochs - 1 if name == "latest.pth.tar" else int(name[2:-8])
        inspection = checkpoint_epoch_reader(path)
        if isinstance(inspection, Mapping):
            recorded_epoch = int(inspection.get("epoch", -1))
            all_tensors_finite = bool(inspection.get("all_tensors_finite", False))
            if not all_tensors_finite:
                raise RuntimeError(f"checkpoint {name} contains a non-finite model tensor")
        else:
            recorded_epoch = int(inspection)
            all_tensors_finite = True
        if recorded_epoch != expected_epoch:
            raise RuntimeError(
                f"checkpoint {name} records epoch {recorded_epoch}, expected {expected_epoch}"
            )
        verified[name] = {
            "bytes": size,
            "sha256": digest,
            "recorded_epoch": recorded_epoch,
            "all_tensors_finite": all_tensors_finite,
        }
    if verified["latest.pth.tar"]["sha256"] != verified[f"e-{epochs - 1}.pth.tar"]["sha256"]:
        raise RuntimeError("latest checkpoint does not match the final epoch checkpoint")
    return verified


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
