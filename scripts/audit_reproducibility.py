#!/usr/bin/env python
"""Audit required reproducibility scaffolding."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_PATHS = [
    "README.md",
    "SUMMARY.md",
    "configs/resource/cpu_vps.yaml",
    "docs/EXPERIMENT_REGISTRY.md",
    "docs/RESULTS.md",
    "data/manifests/datasets.yaml",
    "papers/references.bib",
]


def main() -> int:
    missing = [path for path in REQUIRED_PATHS if not Path(path).exists()]
    payload = {
        "status": "SMOKE_VALIDATED" if not missing else "BLOCKED",
        "missing": missing,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
