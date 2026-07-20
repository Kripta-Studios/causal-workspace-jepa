"""Resource detection and guards for reproducible experiments."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal_workspace_jepa.common.config import get_nested, load_config

BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class ResourceReport:
    profile_name: str
    root: Path
    free_gb: float
    total_gb: float
    cpu_count: int
    min_free_gb: float
    gpu_required: bool
    ok: bool
    messages: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "root": str(self.root),
            "free_gb": round(self.free_gb, 3),
            "total_gb": round(self.total_gb, 3),
            "cpu_count": self.cpu_count,
            "min_free_gb": self.min_free_gb,
            "gpu_required": self.gpu_required,
            "ok": self.ok,
            "messages": list(self.messages),
        }


def inspect_resources(profile_path: str | Path, root: str | Path = ".") -> ResourceReport:
    """Inspect local resources against a profile without allocating large files."""

    config = load_config(profile_path)
    root_path = Path(root).resolve()
    usage = shutil.disk_usage(root_path)
    free_gb = usage.free / BYTES_PER_GB
    total_gb = usage.total / BYTES_PER_GB
    min_free_gb = float(get_nested(config, "storage.min_free_gb", 4))
    gpu_required = bool(get_nested(config, "hardware.gpu_required", False))
    messages: list[str] = []
    if free_gb < min_free_gb:
        messages.append(
            f"BLOCKED_RESOURCE: free disk {free_gb:.2f} GB is below required "
            f"{min_free_gb:.2f} GB"
        )
    if gpu_required and not _has_nvidia_gpu():
        messages.append("BLOCKED_RESOURCE: profile requires GPU but no NVIDIA GPU was detected")
    if not messages:
        messages.append("SMOKE_VALIDATED: resource guard passed")
    return ResourceReport(
        profile_name=str(config.get("name", Path(profile_path).stem)),
        root=root_path,
        free_gb=free_gb,
        total_gb=total_gb,
        cpu_count=os.cpu_count() or 1,
        min_free_gb=min_free_gb,
        gpu_required=gpu_required,
        ok=not any(message.startswith("BLOCKED") for message in messages),
        messages=tuple(messages),
    )


def require_free_disk(profile_path: str | Path, root: str | Path = ".") -> ResourceReport:
    """Raise ``RuntimeError`` if the configured free-disk guard fails."""

    report = inspect_resources(profile_path, root)
    if not report.ok:
        raise RuntimeError("; ".join(report.messages))
    return report


def estimate_activation_bytes(
    *,
    examples: int,
    layers: int,
    positions: int,
    hidden_size: int,
    bytes_per_value: int,
    overhead_fraction: float = 0.15,
) -> int:
    """Estimate activation storage including metadata/checkpoint overhead."""

    base = examples * layers * positions * hidden_size * bytes_per_value
    return int(base * (1.0 + overhead_fraction))


def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None
