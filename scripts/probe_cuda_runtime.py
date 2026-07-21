#!/usr/bin/env python
"""Standalone CUDA architecture/kernel probe for isolated PyTorch runtimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    import torch

    torch.manual_seed(0)
    caught_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        available = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if available else None
        capability = list(torch.cuda.get_device_capability(0)) if available else None
        arch_list = list(torch.cuda.get_arch_list()) if available else []
    caught_messages.extend(str(item.message) for item in caught)

    checks = {}
    operations = {
        "matmul": lambda: torch.randn(16, 16, device="cuda")
        @ torch.randn(16, 16, device="cuda"),
        "conv2d": lambda: torch.nn.Conv2d(2, 4, 3).cuda()(
            torch.randn(1, 2, 8, 8, device="cuda")
        ),
        "gru": lambda: torch.nn.GRU(2, 8).cuda()(
            torch.randn(3, 1, 2, device="cuda"),
            torch.randn(1, 1, 8, device="cuda"),
        )[0],
    }
    if available:
        for name, operation in operations.items():
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    value = operation()
                    torch.cuda.synchronize()
                operation_warnings = [str(item.message) for item in caught]
                caught_messages.extend(operation_warnings)
                checks[name] = {
                    "passed": True,
                    "finite": bool(torch.isfinite(value).all().item()),
                    "shape": list(value.shape),
                    "warnings": operation_warnings,
                }
            except Exception as exc:  # diagnostics must serialize expected kernel failures
                checks[name] = {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
    payload = {
        "label": args.label,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "device": device,
        "capability": capability,
        "arch_list": arch_list,
        "checks": checks,
        "warnings": sorted(set(caught_messages)),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
