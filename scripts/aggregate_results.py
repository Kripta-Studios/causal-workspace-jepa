#!/usr/bin/env python
"""Aggregate metrics tables."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    metrics_dir = Path("artifacts/metrics")
    files = sorted(path.name for path in metrics_dir.glob("*.json"))
    print(json.dumps({"metrics_files": files}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
