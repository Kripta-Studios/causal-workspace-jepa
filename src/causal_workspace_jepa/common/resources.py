"""Resource detection and guards for reproducible experiments."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
    ram_gb: float | None
    gpu_name: str | None
    gpu_vram_gb: float | None
    torch_version: str | None
    torch_cuda_version: str | None
    torch_cuda_available: bool | None
    bf16_supported: bool | None
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
            "ram_gb": None if self.ram_gb is None else round(self.ram_gb, 3),
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": (
                None if self.gpu_vram_gb is None else round(self.gpu_vram_gb, 3)
            ),
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "torch_cuda_available": self.torch_cuda_available,
            "bf16_supported": self.bf16_supported,
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
    ram_gb = _physical_ram_gb()
    gpu = _inspect_nvidia_gpu() if gpu_required else {}
    torch_runtime = _inspect_torch_cuda() if gpu_required else {}
    messages: list[str] = []
    if free_gb < min_free_gb:
        messages.append(
            f"BLOCKED_RESOURCE: free disk {free_gb:.2f} GB is below required "
            f"{min_free_gb:.2f} GB"
        )
    if gpu_required and not gpu:
        messages.append("BLOCKED_RESOURCE: profile requires GPU but no NVIDIA GPU was detected")
    if gpu_required and gpu:
        min_vram_gb = float(get_nested(config, "hardware.min_vram_gb", 0))
        # NVIDIA reports MiB while consumer GPU names use decimal GB. Compare in
        # decimal GB so a 12,227 MiB "12 GB" device satisfies a 12 GB profile.
        if float(gpu["vram_gb"]) < min_vram_gb:
            messages.append(
                f"BLOCKED_RESOURCE: GPU VRAM {float(gpu['vram_gb']):.2f} GB is below "
                f"required {min_vram_gb:.2f} GB"
            )
    if gpu_required and torch_runtime.get("cuda_available") is not True:
        messages.append("BLOCKED_RESOURCE: CUDA-enabled PyTorch runtime is unavailable")
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
        ram_gb=ram_gb,
        gpu_name=str(gpu["name"]) if gpu else None,
        gpu_vram_gb=float(gpu["vram_gb"]) if gpu else None,
        torch_version=(
            str(torch_runtime["torch_version"]) if torch_runtime.get("torch_version") else None
        ),
        torch_cuda_version=(
            str(torch_runtime["cuda_version"]) if torch_runtime.get("cuda_version") else None
        ),
        torch_cuda_available=(
            bool(torch_runtime["cuda_available"])
            if "cuda_available" in torch_runtime
            else None
        ),
        bf16_supported=(
            bool(torch_runtime["bf16_supported"])
            if "bf16_supported" in torch_runtime
            else None
        ),
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


def _inspect_nvidia_gpu() -> dict[str, str | float]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first = result.stdout.strip().splitlines()[0]
        name, memory_mib = (part.strip() for part in first.split(",", maxsplit=1))
        return {"name": name, "vram_gb": float(memory_mib) * 1024**2 / 1e9}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {}


def _inspect_torch_cuda() -> dict[str, str | bool | None]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}
    available = bool(torch.cuda.is_available())
    bf16 = bool(torch.cuda.is_bf16_supported()) if available else False
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": available,
        "bf16_supported": bf16,
    }


def _physical_ram_gb() -> float | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.total_physical) / BYTES_PER_GB
        except (AttributeError, OSError):
            return None
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size / BYTES_PER_GB
    except (AttributeError, OSError, ValueError):
        return None
