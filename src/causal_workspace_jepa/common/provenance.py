"""Experiment provenance helpers."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Provenance:
    time_utc: str
    git_commit: str
    git_dirty: bool
    python: str
    platform: str
    command: str
    seed: int | None
    resource_profile: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def is_git_dirty() -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--short"], text=True)
        return bool(status.strip())
    except Exception:
        return True


def collect_provenance(command: str, resource_profile: str, seed: int | None = None) -> Provenance:
    return Provenance(
        time_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=current_git_commit(),
        git_dirty=is_git_dirty(),
        python=platform.python_version(),
        platform=platform.platform(),
        command=command,
        seed=seed,
        resource_profile=resource_profile,
    )


def write_provenance(path: str | Path, provenance: Provenance, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    payload = json.loads(provenance.to_json())
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
