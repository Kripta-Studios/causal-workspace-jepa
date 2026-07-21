#!/usr/bin/env python
"""Report exact-runtime blockers for the pinned official EB-JEPA checkout."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import platform

from causal_workspace_jepa.adapters.eb_jepa_adapter import verify_eb_jepa_checkout
from causal_workspace_jepa.common.resources import inspect_resources


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def main() -> int:
    modules = (
        "einops",
        "fire",
        "huggingface_hub",
        "matplotlib",
        "cv2",
        "sklearn",
        "tiktoken",
        "torch",
        "torchcodec",
        "torchvision",
        "wandb",
        "gymnasium",
        "imageio",
        "seaborn",
        "submitit",
        "tqdm",
        "pymunk",
        "decord",
        "omegaconf",
        "ruamel.yaml",
    )
    dependencies = {module: _available(module) for module in modules}
    try:
        source = dict(verify_eb_jepa_checkout(Path(".cache/upstream/eb_jepa")))
    except (FileNotFoundError, RuntimeError) as exc:
        source = {"error": str(exc)}
    try:
        import torch

        torch_version = str(torch.__version__)
        capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        arch_list = list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else []
    except ImportError:
        torch_version = None
        capability = None
        arch_list = []
    python_version = platform.python_version()
    exact_runtime = python_version.startswith("3.12.") and bool(
        torch_version and torch_version.startswith("2.6.")
    )
    compatible_runtime = (
        python_version.startswith("3.12.")
        and _version_prefix(torch_version) >= (2, 7)
        and capability == [12, 0]
        and "sm_120" in arch_list
    )
    missing = [module for module, available in dependencies.items() if not available]
    if "error" in source:
        status = "BLOCKED_OFFICIAL_ENV"
    elif exact_runtime and capability == [12, 0] and "sm_120" not in arch_list:
        status = "BLOCKED_GPU_SM120_EXACT_PIN"
    elif missing:
        status = "BLOCKED_OFFICIAL_ENV"
    elif compatible_runtime:
        status = "READY_COMPAT_GPU"
    elif exact_runtime:
        status = "READY_OFFICIAL_ENV"
    else:
        status = "BLOCKED_OFFICIAL_ENV"
    payload = {
        "status": status,
        "source": source,
        "runtime": {
            "python": python_version,
            "torch": torch_version,
            "cuda_compute_capability": capability,
            "compiled_cuda_architectures": arch_list,
            "required_python": "3.12.*",
            "required_torch": "2.6.0",
            "exact": exact_runtime,
            "sm120_compatible_deviation": compatible_runtime,
        },
        "dependencies": dependencies,
        "missing_dependencies": missing,
        "resources": inspect_resources("configs/resource/eb_jepa_official.yaml").as_dict(),
        "note": (
            "Torch 2.6/cu126 is not GPU-executable on SM120. READY_COMPAT_GPU is a declared "
            "dependency deviation and does not imply exact upstream environment reproduction."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"READY_OFFICIAL_ENV", "READY_COMPAT_GPU"} else 2


def _version_prefix(value: str | None) -> tuple[int, int]:
    if value is None:
        return (0, 0)
    try:
        major, minor, *_ = value.split("+", 1)[0].split(".")
        return (int(major), int(minor))
    except (TypeError, ValueError):
        return (0, 0)


if __name__ == "__main__":
    raise SystemExit(main())
