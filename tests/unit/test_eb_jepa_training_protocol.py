import hashlib
from pathlib import Path

import pytest

from causal_workspace_jepa.experiments.world_model.eb_jepa_training_protocol import (
    EXPECTED_SOURCE_CONFIG,
    required_checkpoint_names,
    validate_completed_training_status,
    validate_source_training_config,
)


def _nested_config() -> dict:
    root: dict = {}
    for path, value in EXPECTED_SOURCE_CONFIG.items():
        cursor = root
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def test_training_protocol_accepts_exact_source_config() -> None:
    assert validate_source_training_config(_nested_config()) == EXPECTED_SOURCE_CONFIG


def test_training_protocol_rejects_batch_drift() -> None:
    config = _nested_config()
    config["data"]["batch_size"] = 128
    with pytest.raises(ValueError, match="data.batch_size"):
        validate_source_training_config(config)


def test_training_protocol_freezes_all_and_last_checkpoints() -> None:
    names = required_checkpoint_names(12)
    assert len(names) == 13
    assert names[:2] == ["e-0.pth.tar", "e-1.pth.tar"]
    assert names[-4:] == ["e-9.pth.tar", "e-10.pth.tar", "e-11.pth.tar", "latest.pth.tar"]


def test_completed_training_status_revalidates_bytes_hashes_and_epochs(tmp_path: Path) -> None:
    manifest = {}
    epochs_by_path = {}
    for epoch in range(3):
        path = tmp_path / f"e-{epoch}.pth.tar"
        path.write_bytes(f"checkpoint-{epoch}".encode())
        epochs_by_path[path] = epoch
        manifest[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    latest = tmp_path / "latest.pth.tar"
    latest.write_bytes((tmp_path / "e-2.pth.tar").read_bytes())
    epochs_by_path[latest] = 2
    manifest[latest.name] = {
        "bytes": latest.stat().st_size,
        "sha256": hashlib.sha256(latest.read_bytes()).hexdigest(),
    }
    source_config = {"optim.epochs": 3}
    status = {
        "status": "COMPLETED",
        "experiment_id": "train",
        "seed": 1,
        "source_revision": "abc",
        "source_config": source_config,
        "repo_dirty_at_start": False,
        "source_clean": True,
        "checkpoint_manifest": manifest,
    }
    verified = validate_completed_training_status(
        status,
        tmp_path,
        experiment_id="train",
        seed=1,
        source_revision="abc",
        source_config=source_config,
        epochs=3,
        checkpoint_epoch_reader=epochs_by_path.__getitem__,
    )
    assert verified["e-2.pth.tar"]["recorded_epoch"] == 2
    (tmp_path / "e-1.pth.tar").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="bytes/hash"):
        validate_completed_training_status(
            status,
            tmp_path,
            experiment_id="train",
            seed=1,
            source_revision="abc",
            source_config=source_config,
            epochs=3,
            checkpoint_epoch_reader=epochs_by_path.__getitem__,
        )


def test_completed_training_status_rejects_nonfinite_checkpoint_inspection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "e-0.pth.tar"
    latest = tmp_path / "latest.pth.tar"
    path.write_bytes(b"checkpoint")
    latest.write_bytes(b"checkpoint")
    record = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    status = {
        "status": "COMPLETED",
        "experiment_id": "train",
        "seed": 1,
        "source_revision": "abc",
        "source_config": {},
        "repo_dirty_at_start": False,
        "source_clean": True,
        "checkpoint_manifest": {"e-0.pth.tar": record, "latest.pth.tar": record},
    }
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_completed_training_status(
            status,
            tmp_path,
            experiment_id="train",
            seed=1,
            source_revision="abc",
            source_config={},
            epochs=1,
            checkpoint_epoch_reader=lambda _path: {
                "epoch": 0,
                "all_tensors_finite": False,
            },
        )
