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
    except ImportError:
        torch_version = None
        capability = None
    python_version = platform.python_version()
    exact_runtime = python_version.startswith("3.12.") and bool(
        torch_version and torch_version.startswith("2.6.")
    )
    missing = [module for module, available in dependencies.items() if not available]
    payload = {
        "status": (
            "READY_OFFICIAL_ENV"
            if exact_runtime and not missing and "error" not in source
            else "BLOCKED_OFFICIAL_ENV"
        ),
        "source": source,
        "runtime": {
            "python": python_version,
            "torch": torch_version,
            "cuda_compute_capability": capability,
            "required_python": "3.12.*",
            "required_torch": "2.6.0",
            "exact": exact_runtime,
        },
        "dependencies": dependencies,
        "missing_dependencies": missing,
        "resources": inspect_resources("configs/resource/eb_jepa_official.yaml").as_dict(),
        "note": (
            "A failed exact-runtime check does not invalidate the separate source-contract smoke; "
            "it blocks a claim of exact upstream training reproduction."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "READY_OFFICIAL_ENV" else 2


if __name__ == "__main__":
    raise SystemExit(main())
