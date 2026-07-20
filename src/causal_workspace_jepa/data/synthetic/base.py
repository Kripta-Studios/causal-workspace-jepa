"""Shared helpers for deterministic Tier 0 synthetic data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SyntheticDataset:
    name: str
    observations: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    masks: np.ndarray | None = None

    @property
    def transitions(self) -> int:
        return int(self.actions.shape[0] * self.actions.shape[1])

    def arrays(self) -> dict[str, np.ndarray]:
        payload = {
            "observations": self.observations,
            "actions": self.actions,
            "states": self.states,
            "metadata_json": np.array(json.dumps(self.metadata, sort_keys=True)),
        }
        if self.masks is not None:
            payload["masks"] = self.masks
        return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_dataset(dataset: SyntheticDataset, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{dataset.name}.npz"
    np.savez_compressed(path, **dataset.arrays())
    stat = path.stat()
    return {
        "name": dataset.name,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": stat.st_size,
        "observations_shape": list(dataset.observations.shape),
        "actions_shape": list(dataset.actions.shape),
        "states_shape": list(dataset.states.shape),
        "transitions": dataset.transitions,
        "metadata": dataset.metadata,
    }


def load_dataset(path: str | Path) -> SyntheticDataset:
    loaded = np.load(path, allow_pickle=False)
    metadata = json.loads(str(loaded["metadata_json"]))
    masks = loaded["masks"] if "masks" in loaded.files else None
    return SyntheticDataset(
        name=Path(path).stem,
        observations=loaded["observations"],
        actions=loaded["actions"],
        states=loaded["states"],
        metadata=metadata,
        masks=masks,
    )
