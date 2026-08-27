"""Experiment provenance helpers."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


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


def stage_cli_command(
    module: str,
    stage: str,
    output: str,
    require_development: str | None = None,
    extra_args: str = "",
) -> str:
    """Build a single-stage CLI string. Do not fuse development and confirmation.

    Confirmation may include ``--require-development`` (authorization input).
    It must not include ``--stage development`` or a fused ``&&`` second stage.
    Optional ``extra_args`` (for example ``--rung 200``) must not add a second stage.
    """

    if stage not in {"development", "confirmation"}:
        raise ValueError("stage must be development or confirmation")
    command = f"python -m {module} --stage {stage} --output {output}"
    if extra_args:
        extra = extra_args.strip()
        if "--stage " in extra:
            raise ValueError("extra_args must not contain --stage")
        if "&&" in extra:
            raise ValueError("extra_args must not fuse commands")
        command += f" {extra}"
    if stage == "confirmation":
        auth = require_development or "artifacts/metrics/crct_jepa_action_delta_v1.dev.json"
        command += f" --require-development {auth}"
    return command


def write_stage_provenance(
    path: str | Path,
    *,
    module: str,
    stage: str,
    output: str,
    experiment_id: str,
    seeds: Sequence[int],
    resource_profile: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write per-stage provenance. Confirmation must not inherit development seeds."""

    if not seeds:
        raise ValueError("seeds must be non-empty")
    require_development = None
    if extra and "require_development" in extra:
        require_development = str(extra["require_development"])
    command = stage_cli_command(module, stage, output, require_development=require_development)
    provenance = collect_provenance(command, resource_profile, seed=int(seeds[0]))
    payload = {
        "experiment_id": experiment_id,
        "stage": stage,
        "seeds": [int(s) for s in seeds],
        "metrics": Path(output).as_posix(),
        "command_stage": stage,
    }
    if extra:
        payload.update(extra)
    write_provenance(path, provenance, extra=payload)


def write_provenance(path: str | Path, provenance: Provenance, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    payload = json.loads(provenance.to_json())
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
